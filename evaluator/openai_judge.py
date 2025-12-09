import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from evaluator.eval import TextOnlyRAGEvaluator

logger = logging.getLogger(__name__)
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

@dataclass
class OpenAIJudgeConfig:
    model: str = "gpt-4o-mini"
    output_dir: str = "evaluator/evaluation_results"
    api_key: Optional[str] = None
    temperature: float = 0.0
    top_k_contexts: int = 3
    context_char_limit: int = 600
    system_prompt: str = (
        "You are an impartial evaluator for Retrieval-Augmented Generation systems. "
        "Given a query, optional gold references, and retrieved context snippets, "
        "score whether the snippets allow answering the question faithfully. "
        "Respond strictly in JSON as instructed."
    )

class OpenAIJudge:

    def __init__(self, evaluator: TextOnlyRAGEvaluator, config: OpenAIJudgeConfig) -> None:
        self.evaluator = evaluator
        self.config = config
        key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Export it before running.")
        self.client = OpenAI(api_key=key)

    def run(self) -> pd.DataFrame:
        os.makedirs(self.config.output_dir, exist_ok=True)
        all_qs = sorted(
            set(self.evaluator.gold_lookup.keys()) | set(self.evaluator.results_map.keys())
        )
        rows = [self.judge_query(qk) for qk in all_qs]
        df = pd.DataFrame(rows)
        per_query = os.path.join(self.config.output_dir, "openai_hybrid.csv")
        df.to_csv(per_query, index=False)
        summary_path = os.path.join(self.config.output_dir, "openai_summary_hybrid.json")
        summary = self._summarize(df)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return df

    def judge_query(self, canon_key: str) -> Dict[str, Any]:
        question = self.evaluator.display_lookup.get(canon_key, canon_key)
        results_entry = self.evaluator.results_map.get(canon_key, {"texts": []})
        retrieved_texts: List[str] = results_entry.get("texts", [])
        subset = retrieved_texts[: self.config.top_k_contexts]

        if not subset:
            return self._record(
                canon_key, question, subset, verdict="insufficient", precision=0.0, confidence=0.0, error="no_context"
            )

        prompt = self._build_prompt(question, [], subset)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            cleaned = self._strip_code_fence(content)
            parsed = json.loads(cleaned)
            return self._record(
                canon_key,
                question,
                subset,
                verdict=parsed.get("verdict", "undecided"),
                precision=float(parsed.get("context_precision", 0.0)),
                confidence=float(parsed.get("confidence", 0.0)),
                raw_response=content,
            )
        except Exception as err:
            return self._record(
                canon_key, question, subset, verdict="error", precision=0.0, confidence=0.0, error=str(err)
            )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = [ln for ln in stripped.splitlines() if not ln.strip().startswith("```")]
            if lines and lines[0].lower().startswith("json"):
                lines = lines[1:]
            return "\n".join(lines).strip()
        return stripped

    def _build_prompt(
        self, question: str, gold_texts: Sequence[str], retrieved_texts: Sequence[str]
    ) -> str:
        gold_str = self._join_snippets(gold_texts, "No reference available.")
        ctx_str = self._join_snippets(retrieved_texts, "No retrieved context.")
        instructions = (
            "Return JSON with fields: answer_coverage (0-1), context_precision (0-1), "
            "confidence (0-1), verdict ('sufficient' or 'insufficient'), rationale (short text)."
        )
        return (
            f"{instructions}\n\n"
            f"Question:\n{question}\n\n"
            f"Gold Reference Snippets:\n{gold_str}\n\n"
            f"Retrieved Context Snippets:\n{ctx_str}\n"
        )

    def _join_snippets(self, snippets: Sequence[str], placeholder: str) -> str:
        if not snippets:
            return placeholder
        formatted = []
        limit = max(self.config.context_char_limit, 0)
        for idx, snippet in enumerate(snippets, 1):
            cleaned = " ".join(str(snippet).strip().split())
            if limit and len(cleaned) > limit:
                cleaned = cleaned[:limit].rstrip() + "..."
            formatted.append(f"{idx}. {cleaned}")
        return "\n".join(formatted)

    def _record(
        self,
        canon_key: str,
        question: str,
        retrieved_subset: Sequence[str],
        verdict: str,
        precision: float = 0.0,
        confidence: float = 0.0,
        raw_response: str = "",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "query": question,
            "canon_key": canon_key,
            "split": self.evaluator.query_split.get(canon_key, "unknown"),
            "verdict": verdict,
            "precision_score": precision,
            "confidence": confidence,
            "raw_response": raw_response,
            "error": error,
            "model": self.config.model,
            "num_contexts": len(list(retrieved_subset)),
            "context_preview": self._preview_snippets(retrieved_subset),
        }

    @staticmethod
    def _preview_snippets(snippets: Sequence[str]) -> str:
        return " || ".join(" ".join(str(s).split()) for s in snippets if str(s).strip())

    def _summarize(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"total_queries": 0, "model": self.config.model}
        verdicts = df["verdict"].value_counts().to_dict()
        return {
            "model": self.config.model,
            "total_queries": int(df.shape[0]),
            "verdict_breakdown": verdicts,
            "mean_precision_score": float(df["precision_score"].fillna(0.0).mean()),
            "mean_confidence": float(df["confidence"].fillna(0.0).mean()),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenAI LLM judge.")
    parser.add_argument(
        "--multi-csv",
        default="queries/multi_file_retrieval_queries.csv",
        help="Path to the multi-file gold CSV.",
    )
    parser.add_argument(
        "--single-csv",
        default="queries/single_file_retrieval_queries.csv",
        help="Path to the single-file gold CSV.",
    )
    parser.add_argument(
        "--results-json",
        default="results/rag_results.json",
        help="Path to the retrieval results JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--context-topk",
        type=int,
        default=3,
        help="Number of retrieved chunks per prompt.",
    )
    parser.add_argument(
        "--context-char-limit",
        type=int,
        default=600,
        help="Character limit per snippet in prompt.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    evaluator = TextOnlyRAGEvaluator(
        multi_csv=args.multi_csv,
        single_csv=args.single_csv,
        results_json=args.results_json,
        out_dir=args.out_dir,
    )
    config = OpenAIJudgeConfig(
        model="gpt-4o-mini",
        output_dir=args.out_dir,
        api_key=None,
        temperature=args.temperature,
        top_k_contexts=args.context_topk,
        context_char_limit=args.context_char_limit,
    )
    judge = OpenAIJudge(evaluator, config)
    judge.run()


if __name__ == "__main__":
    main()

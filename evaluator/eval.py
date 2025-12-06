import os
import json
import ast
import re
import string
import unicodedata
from math import log2
from typing import List, Dict, Tuple
import difflib

import numpy as np
import pandas as pd

# -------------------- utilities --------------------

def _exists_any(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the columns {candidates} found. Available: {list(df.columns)}")

# -------------------- evaluator --------------------

class TextOnlyRAGEvaluator:
    """
    Text-only evaluator for RAG retrieval where gold supervision is *only* gold-text snippets.

    Ground truth CSVs (both are used and merged by query):
      - multi_file_retrieval_queries.csv
      - single_file_retrieval_queries.csv  (or singlefile_retrieval_queries.csv)

    Required columns (auto-detected names):
      - query
      - gold-text (or gold_text / gold / goldtext)  -> stored as raw_gold-text in outputs

    Retrieval results JSON:
      - rag_results.json with per-query 'results[*].match' (or 'text') and optional 'filename'.
    """

    _SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+')
    _STOP = set("""
        a an the and or of in on to for from with as at by is are was were be been being
        this that these those it its their his her we you they i vs v et al
    """.split())
    _EXT_RE = re.compile(r'\.(pdf|txt|md|docx|pptx|html|htm|csv|json)$', re.I)

    # ---------------- canonicalization ----------------
    def _canon_query(self, s: str) -> str:
        _PUNCT_TO_STRIP = string.punctuation + "“”‘’´`•·•–—-"  # add common unicode punct/dashes
        _ZWSP_LIKE = "".join([
            "\u200b",  # zero width space
            "\u200c",  # zero width non-joiner
            "\u200d",  # zero width joiner
            "\ufeff",  # zero width no-break space (BOM)
            "\u00a0",  # non-breaking space
        ])
        # 1) Unicode normalize
        s = unicodedata.normalize("NFKC", str(s))

        # 2) Remove zero-width / NBSPs
        s = s.translate({ord(c): None for c in _ZWSP_LIKE})

        # 3) Normalize fancy quotes/dashes first, then strip all punctuation
        s = (s.replace("“", '"').replace("”", '"')
              .replace("‘", "'").replace("’", "'")
              .replace("–", "-").replace("—", "-").replace("-", "-"))
        s = s.translate(str.maketrans("", "", _PUNCT_TO_STRIP))

        # 4) Lowercase and collapse whitespace
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def __init__(
        self,
        multi_csv: str,
        single_csv: str,
        results_json: str,
        out_dir: str = "./eval_out",
        topk: List[int] = (1, 3, 5, 10),
        query_col_candidates: List[str] = ("query", "question"),
        gold_text_col_candidates: List[str] = ("gold-text", "gold_text", "gold", "goldtext", "gold-doc"),
    ):
        # Allow path variants for single_csv
        single_csv = _exists_any([single_csv, single_csv.replace("single_file_", "singlefile_"),
                                  single_csv.replace("singlefile_", "single_file_")]) or single_csv

        if not os.path.exists(multi_csv):
            raise FileNotFoundError(f"multi_csv not found: {multi_csv}")
        if not os.path.exists(single_csv):
            print(f"[warn] single_csv not found: {single_csv} — continuing with multi only.")

        self.multi_csv = multi_csv
        self.single_csv = single_csv if os.path.exists(single_csv) else ""
        self.results_json = results_json
        self.out_dir = out_dir
        self.topk = list(topk)

        os.makedirs(self.out_dir, exist_ok=True)

        # Load ground truth CSVs
        self.multi_df = pd.read_csv(self.multi_csv)
        if self.single_csv:
            self.single_df = pd.read_csv(self.single_csv)
        else:
            self.single_df = pd.DataFrame(columns=list(self.multi_df.columns))

        # Detect column names
        self.query_col = _pick_col(self.multi_df if len(self.multi_df) else self.single_df, list(query_col_candidates))
        self.gold_text_col = _pick_col(self.multi_df if len(self.multi_df) else self.single_df, list(gold_text_col_candidates))

        # If the other split uses different casing, fix too
        for df in (self.multi_df, self.single_df):
            if self.query_col not in df.columns and len(df.columns):
                alt = _pick_col(df, list(query_col_candidates))
                df.rename(columns={alt: self.query_col}, inplace=True)
            if self.gold_text_col not in df.columns and len(df.columns):
                alt = _pick_col(df, list(gold_text_col_candidates))
                df.rename(columns={alt: self.gold_text_col}, inplace=True)

        # Load results
        with open(self.results_json, "r", encoding="utf-8") as f:
            self.results_data = json.load(f)

        # Build gold lookup and query->split map (keys are canonicalized)
        self.gold_lookup, self.query_split, self.display_lookup = self._build_gold_textonly()

        # Build results map (canonicalized) and merge display names
        self.results_map, results_display = self._build_results_map()
        # Prefer gold display text when available; otherwise use results' original text
        for k, disp in results_display.items():
            self.display_lookup.setdefault(k, disp)

    # --------------- parsing helpers ---------------
    def _chunk_overlap_ratio(self, rt: str, gold_texts: List[str]) -> float:
        """
        Portion of a retrieved chunk's characters that overlap with the BEST-matching gold snippet.
        Uses character-level alignment (SequenceMatcher) on normalized strings.
        Returns a value in [0, 1].
        """
        rt_n = self._normalize_text(rt)
        if not rt_n:
            return 0.0

        best = 0.0
        for gt in gold_texts:
            gt_n = self._normalize_text(gt)
            if not gt_n:
                continue
            sm = difflib.SequenceMatcher(None, rt_n, gt_n, autojunk=False)
            # Sum sizes for all exact-matching blocks
            equal_chars = sum(m.size for m in sm.get_matching_blocks())
            # Note: get_matching_blocks() ends with a size-0 sentinel—adds 0, safe to include.
            ratio = equal_chars / max(1, len(rt_n))
            if ratio > best:
                best = ratio
        return best


    def chunk_overlap_metrics(self, retrieved_texts: List[str], gold_texts: List[str]) -> Dict[str, float]:
        """
        Computes the average per-chunk overlap ratio across all retrieved chunks for a query.
        Returns:
          - chunk_overlap: mean fraction of retrieved chunk content that overlaps with gold
          - min_chunk_overlap / max_chunk_overlap: extremes across retrieved chunks
          - chunk_overlap_count: number of retrieved chunks considered
        """
        r_clean = [r for r in retrieved_texts if str(r).strip()]
        g_clean = [g for g in gold_texts if str(g).strip()]

        if not r_clean:
            return {
                'chunk_overlap': 0.0,
                'min_chunk_overlap': 0.0,
                'max_chunk_overlap': 0.0,
                'chunk_overlap_count': 0,
            }

        overlaps = [self._chunk_overlap_ratio(rt, g_clean) for rt in r_clean]
        return {
            'chunk_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
            'min_chunk_overlap': float(np.min(overlaps)) if overlaps else 0.0,
            'max_chunk_overlap': float(np.max(overlaps)) if overlaps else 0.0,
            'chunk_overlap_count': int(len(overlaps)),
        }


    def _parse_gold_texts(self, v) -> List[str]:
        """
        Parse gold-text into a list. We do NOT split on commas.
        Accepts: Python/JSON list/tuple, or '||' / ';' delimiters; else a single string.
        """
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        s = str(v).strip()
        if not s:
            return []
        # Python/JSON list-like
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if str(x).strip()]
                return [str(parsed)]
            except Exception:
                pass
        # Explicit delimiters
        if '||' in s:
            return [t.strip() for t in s.split('||') if t.strip()]
        if ';' in s:
            return [t.strip() for t in s.split(';') if t.strip()]
        return [s]

    def _build_gold_for_df(self, df: pd.DataFrame) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """
        Returns:
          out: canon_query -> {"texts": [...], "raw_text": str}
          display: canon_query -> first seen original query (for pretty printing)
        """
        out: Dict[str, Dict] = {}
        display: Dict[str, str] = {}
        for _, row in df.iterrows():
            q_orig = str(row[self.query_col])
            q = self._canon_query(q_orig)
            raw_txt = "" if row.get(self.gold_text_col) is None else str(row.get(self.gold_text_col))
            texts = self._parse_gold_texts(row.get(self.gold_text_col, ""))

            if q not in out:
                out[q] = {"texts": [], "raw_text": raw_txt}
                display[q] = q_orig  # preserve a human-readable version
            else:
                if not out[q].get("raw_text"):
                    out[q]["raw_text"] = raw_txt

            out[q]["texts"].extend(texts)

        # Dedup texts while preserving order
        for qk in out:
            seen, uniq = set(), []
            for t in out[qk]["texts"]:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            out[qk]["texts"] = uniq
        return out, display

    def _build_gold_textonly(self) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, str]]:
        gold: Dict[str, Dict] = {}
        split_map: Dict[str, str] = {}
        display: Dict[str, str] = {}

        multi, multi_disp = self._build_gold_for_df(self.multi_df)
        single, single_disp = self._build_gold_for_df(self.single_df)

        for q, v in multi.items():
            gold[q] = {"texts": list(v["texts"]), "raw_text": v.get("raw_text", "")}
            split_map[q] = "multi"
            display[q] = multi_disp.get(q, q)

        for q, v in single.items():
            if q in gold:
                # merge texts (dedup)
                for t in v["texts"]:
                    if t not in gold[q]["texts"]:
                        gold[q]["texts"].append(t)
                if not gold[q].get("raw_text"):
                    gold[q]["raw_text"] = v.get("raw_text", "")
                split_map[q] = "both"
                # keep existing display (prefer multi), or set if missing
                display.setdefault(q, single_disp.get(q, q))
            else:
                gold[q] = {"texts": list(v["texts"]), "raw_text": v.get("raw_text", "")}
                split_map[q] = "single"
                display[q] = single_disp.get(q, q)

        return gold, split_map, display

    # --------------- results mapping ---------------

    def _build_results_map(self) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, str]]:
        mapping: Dict[str, Dict[str, List[str]]] = {}
        display: Dict[str, str] = {}
        for item in self.results_data.get("queries", []):
            q_orig = str(item.get("query", ""))
            q = self._canon_query(q_orig)
            docs, texts = [], []
            for r in item.get("results", []):
                fn = r.get("filename") or r.get("doc") or r.get("document") or r.get("id") or ""
                txt = r.get("match") or r.get("text") or ""
                if fn:
                    docs.append(str(fn))
                if txt is not None and str(txt).strip():
                    texts.append(str(txt))
            # Dedup docs keeping order
            seen, uniq_docs = set(), []
            for d in docs:
                if d not in seen:
                    seen.add(d)
                    uniq_docs.append(d)
            mapping[q] = {"docs": uniq_docs, "texts": texts}
            display.setdefault(q, q_orig)
        return mapping, display

    # --------------- normalization / tokenization ---------------

    def _normalize_text(self, s: str) -> str:
        return ' '.join(str(s).lower().split())

    def _tok(self, text: str) -> List[str]:
        t = str(text).lower()
        t = t.translate(str.maketrans('', '', string.punctuation))
        toks = [w for w in t.split() if w and w not in self._STOP]
        return toks

    def _tok_all(self, text: str) -> List[str]:
        """Tokenizer for BLEU (keeps stopwords)."""
        t = str(text).lower()
        t = t.translate(str.maketrans('', '', string.punctuation))
        toks = [w for w in t.split() if w]
        return toks

    def _split_sents(self, text: str) -> List[str]:
        return [s.strip() for s in self._SENT_SPLIT.split(str(text).strip()) if s.strip()]

    # --------------- BLEU (corpus-style over concatenated texts) ---------------

    def _count_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        counts: Dict[Tuple[str, ...], int] = {}
        if len(tokens) < n:
            return counts
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i+n])
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    def bleu_score(self, retrieved_texts: List[str], gold_texts: List[str], max_n: int = 4, smoothing: float = 1e-9) -> float:
        """
        Corpus BLEU between concatenated retrieved_texts (candidate) and gold_texts (reference).
        - Uniform weights over n=1..max_n
        - Modified precision with clipping
        - Brevity penalty
        - Light smoothing to avoid log(0)
        """
        cand = " ".join(str(x) for x in retrieved_texts if str(x).strip())
        ref  = " ".join(str(x) for x in gold_texts      if str(x).strip())
        if not cand or not ref:
            return 0.0

        cand_tok = self._tok_all(cand)
        ref_tok  = self._tok_all(ref)

        c_len = len(cand_tok)
        r_len = len(ref_tok)

        precisions = []
        for n in range(1, max_n + 1):
            cand_counts = self._count_ngrams(cand_tok, n)
            ref_counts  = self._count_ngrams(ref_tok, n)

            if not cand_counts:
                precisions.append(0.0)
                continue

            # clipped counts
            match = 0
            total = 0
            for ng, cnt in cand_counts.items():
                total += cnt
                match += min(cnt, ref_counts.get(ng, 0))

            # smoothing to avoid zero precision nuking the whole score
            p_n = (match + smoothing) / (total + smoothing)
            precisions.append(p_n)

        # Brevity penalty
        if c_len == 0:
            return 0.0
        bp = 1.0 if c_len > r_len else float(np.exp(1 - (r_len / max(1, c_len))))

        # geometric mean of precisions
        gm = float(np.exp(np.mean([np.log(p) for p in precisions])))

        return bp * gm

    # --------------- robust overlap (tiered) ---------------

    def _substring_containment(self, a: str, b: str) -> float:
        """
        Character containment of a in b, after normalization.
        returns |a| / |b| if a is substring of b, else 0.
        We check both directions externally.
        """
        a_n = self._normalize_text(a)
        b_n = self._normalize_text(b)
        if not a_n or not b_n:
            return 0.0
        if a_n in b_n:
            return len(a_n) / max(1, len(b_n))
        return 0.0

    def _char_ngram_containment(self, a: str, b: str, n: int = 5) -> float:
        """
        Character n-gram containment: |ngrams(a) ∩ ngrams(b)| / |ngrams(a)|.
        Helps when wording changes slightly.
        """
        a_n = self._normalize_text(a)
        b_n = self._normalize_text(b)
        if len(a_n) < n or len(b_n) < n:
            return 0.0
        def grams(s): return {s[i:i+n] for i in range(len(s)-n+1)}
        A, B = grams(a_n), grams(b_n)
        if not A:
            return 0.0
        return len(A & B) / len(A)

    def _token_jaccard(self, a: str, b: str) -> float:
        A, B = set(self._tok(a)), set(self._tok(b))
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def _match_score(self, gold: str, cand: str) -> float:
        """
        Tiered score in [0,1]: max over
         - exact/substring containment (both directions)
         - token Jaccard
         - char n-gram containment
        """
        # substring containment (both directions)
        s1 = self._substring_containment(gold, cand)
        s2 = self._substring_containment(cand, gold)
        # token Jaccard
        sj = self._token_jaccard(gold, cand)
        # char n-gram containment
        sc = self._char_ngram_containment(gold, cand, n=5)
        return max(s1, s2, sj, sc)

    # --------------- text metrics ---------------

    def text_overlap_metrics(
        self,
        retrieved_texts: List[str],
        gold_texts: List[str],
        recall_thresh: float = 0.30,     # gentler default
        precision_thresh: float = 0.20,  # gentler default
    ) -> Dict[str, float]:
        """
        Use the tiered _match_score to compute:
          - text_recall: fraction of gold snippets matched by any retrieved
          - text_precision: fraction of retrieved snippets that match some gold
        """
        if not gold_texts or all(not str(gt).strip() for gt in gold_texts):
            return {'text_recall': 0.0, 'text_precision': 0.0, 'text_f1': 0.0,
                    'num_gold_texts_matched': 0}

        g_clean = [g for g in gold_texts if str(g).strip()]
        r_clean = [r for r in retrieved_texts if str(r).strip()]

        # Recall
        found = 0
        for g in g_clean:
            if any(self._match_score(g, r) >= recall_thresh for r in r_clean):
                found += 1
        text_recall = found / len(g_clean) if g_clean else 0.0

        # Precision
        rel_ret = 0
        for r in r_clean:
            if any(self._match_score(r, g) >= precision_thresh for g in g_clean):
                rel_ret += 1
        text_precision = rel_ret / len(r_clean) if r_clean else 0.0

        text_f1 = (2 * text_precision * text_recall / (text_precision + text_recall)
                   if (text_precision + text_recall) > 0 else 0.0)

        return {
            'text_recall': text_recall,
            'text_precision': text_precision,
            'text_f1': text_f1,
            'num_gold_texts_matched': found,
        }

    def context_sentence_metrics(self, gold_texts: List[str], retrieved_texts: List[str], thresh: float = 0.35):
        """Sentence-level context precision/recall/F1 using the same tiered _match_score."""
        gold_sents = [s for t in gold_texts for s in self._split_sents(t)]
        ret_sents = [s for t in retrieved_texts for s in self._split_sents(t)]
        if not gold_sents or not ret_sents:
            return {'ctx_precision': 0.0, 'ctx_recall': 0.0, 'ctx_f1': 0.0}

        matched_gold, matched_ret = set(), set()
        for i, rt in enumerate(ret_sents):
            for j, gt in enumerate(gold_sents):
                if self._match_score(rt, gt) >= thresh:
                    matched_gold.add(j)
                    matched_ret.add(i)
                    break

        prec = len(matched_ret) / len(ret_sents) if ret_sents else 0.0
        rec  = len(matched_gold) / len(gold_sents) if gold_sents else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {'ctx_precision': prec, 'ctx_recall': rec, 'ctx_f1': f1}

   # rank-aware text metrics
    def _text_supported_flags(self, retrieved_texts: List[str], gold_texts: List[str], thresh=0.30) -> List[int]:
        g_clean = [g for g in gold_texts if str(g).strip()]
        r_clean = [r for r in retrieved_texts if str(r).strip()]
        flags = []
        for r in r_clean:
            hit = any(self._match_score(r, g) >= thresh for g in g_clean)
            flags.append(1 if hit else 0)
        return flags

    def _text_recall_at_k(self, retrieved_texts: List[str], gold_texts: List[str], k: int, thresh=0.30) -> float:
        g_clean = [g for g in gold_texts if str(g).strip()]
        r_clean = [r for r in retrieved_texts[:k] if str(r).strip()]
        if not g_clean:
            return 0.0
        found = 0
        for g in g_clean:
            if any(self._match_score(g, r) >= thresh for r in r_clean):
                found += 1
        return float(found / len(g_clean))

    # --------------- per-query evaluation ---------------

    def evaluate_query(self, canon_query_key: str) -> Dict:
        # canon_query_key is already canonicalized in evaluate()
        gold = self.gold_lookup.get(canon_query_key, {"texts": [], "raw_text": ""})
        gtexts: List[str] = gold["texts"]

        res = self.results_map.get(canon_query_key, {"docs": [], "texts": []})
        rdocs: List[str] = res["docs"]
        rtexts: List[str] = res["texts"]

        display_q = self.display_lookup.get(canon_query_key, canon_query_key)

        out = {
            "query": display_q,
            "split": self.query_split.get(canon_query_key, "unknown"),
            "raw_gold-text": gold.get("raw_text", ""),
            "num_gold_texts": len(gtexts),
            "num_retrieved": len(rdocs),
            "retrieved_docs": rtexts[:10],
        }

        # Text overlap (robust)
        out.update(self.text_overlap_metrics(rtexts, gtexts))

        # Sentence-level context metrics
        out.update(self.context_sentence_metrics(gtexts, rtexts))

        # Chunk-level overlap metrics
        out.update(self.chunk_overlap_metrics(rtexts, gtexts))

        # BLEU (corpus-style over concatenated retrieved vs gold)
        out["bleu"] = float(self.bleu_score(rtexts, gtexts))

        # Rank-aware text metrics
        flags = self._text_supported_flags(rtexts, gtexts, thresh=0.30)
        for k in (1, 3):
            if k in self.topk:
                top_flags = flags[:k]
                out[f"text_supported@{k}"] = float(sum(top_flags) / k) if k > 0 and top_flags else 0.0
                out[f"text_recall@{k}"] = self._text_recall_at_k(rtexts, gtexts, k, thresh=0.30)

        return out

    # --------------- evaluate all / aggregate ---------------

    def evaluate(self) -> Tuple[Dict, pd.DataFrame]:
        gold_qs = set(self.gold_lookup.keys())
        res_qs = set(self.results_map.keys())
        all_qs = sorted(list(gold_qs | res_qs))  # canonicalized keys

        rows = [self.evaluate_query(qk) for qk in all_qs]
        df = pd.DataFrame(rows)
        agg = self._aggregate(df)
        return agg, df

    def _aggregate(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {"total_queries": 0}

        agg: Dict[str, float] = {
            "total_queries": int(df.shape[0]),
            "queries_with_multiple_texts": int((df["num_gold_texts"] > 1).sum()),
            "avg_gold_texts_per_query": float(df["num_gold_texts"].mean()),
        }

        to_stat = [
            "text_recall", "text_precision", "text_f1",
            "ctx_precision", "ctx_recall", "ctx_f1",
            "chunk_overlap",                # NEW
            "min_chunk_overlap",            # optional, keep if you want distro stats
            "max_chunk_overlap",            # optional
            *(f"text_supported@{k}" for k in self.topk),
            *(f"text_recall@{k}" for k in self.topk),
            "bleu",                         # BLEU added
        ]

        for m in to_stat:
            vals = df[m].fillna(0.0).astype(float).values if m in df else np.array([0.0])
            agg[f"mean_{m}"] = float(np.mean(vals))
            agg[f"std_{m}"]  = float(np.std(vals))
            agg[f"min_{m}"]  = float(np.min(vals))
            agg[f"max_{m}"]  = float(np.max(vals))

        for split in ["multi", "single", "both"]:
            sdf = df[df["split"] == split]
            agg[f"{split}_queries"] = int(sdf.shape[0])
            for m in ["text_f1", "ctx_f1", "text_recall@3", "text_recall@5"]:
                agg[f"{split}_mean_{m}"] = float(sdf[m].fillna(0.0).mean()) if m in sdf else 0.0
        return agg

    # --------------- outputs ---------------

    def write_outputs(self, agg: Dict, df: pd.DataFrame):
        os.makedirs(self.out_dir, exist_ok=True)
        per_query_csv = os.path.join(self.out_dir, "per_query_metrics.csv")
        df.to_csv(per_query_csv, index=False)

        overall_json = os.path.join(self.out_dir, "summary.json")
        with open(overall_json, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

        report_path = os.path.join(self.out_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("CS RAG (TEXT-ONLY) RETRIEVAL EVALUATION — robust matching\n")
            f.write("=" * 90 + "\n\n")
            f.write("DATASET STATS\n")
            f.write("-" * 90 + "\n")
            f.write(f"Total Queries: {agg['total_queries']}\n")
            f.write(f"Queries with Multiple Gold Texts: {agg['queries_with_multiple_texts']}\n")
            f.write(f"Avg Gold Texts / Query: {agg['avg_gold_texts_per_query']:.2f}\n\n")

            f.write("GLOBAL METRICS (means ± stdev)\n")
            f.write("-" * 90 + "\n")
            for m in ["text_recall", "text_precision", "text_f1",
                "ctx_precision", "ctx_recall", "ctx_f1",
                "chunk_overlap",
                "text_supported@3", "text_recall@3",
                "bleu"]:
                if f"mean_{m}" in agg:
                    f.write(f"{m:>18}: {agg[f'mean_{m}']:.4f} ± {agg[f'std_{m}']:.4f} "
                            f"(min {agg[f'min_{m}']:.4f}, max {agg[f'max_{m}']:.4f})\n")

            f.write("\nSPLIT-SPECIFIC (means)\n")
            f.write("-" * 90 + "\n")
            for split in ["multi", "single", "both"]:
                f.write(f"{split.upper():<6} | n={agg.get(f'{split}_queries',0):<3d} | "
                        f"tF1={agg.get(f'{split}_mean_text_f1',0.0):.4f}  "
                        f"cF1={agg.get(f'{split}_mean_ctx_f1',0.0):.4f}  "
                        f"tR@3={agg.get(f'{split}_mean_text_recall@3',0.0):.4f}\n ")
        print(f"[✓] Wrote:\n - {per_query_csv}\n - {overall_json}\n - {report_path}")

# ------------- Colab-friendly wrapper -------------

def evaluate_cs_rag_textonly(
    multi_csv="../queries/multi_file_retrieval_queries.csv",
    single_csv="../queries/single_file_retrieval_queries.csv",
    results_json="../results/semantic_rag_results.json",
    out_dir="./eval_out",
    topk=(1, 3),
    return_df=True,
):
    ev = TextOnlyRAGEvaluator(
        multi_csv=multi_csv,
        single_csv=single_csv,
        results_json=results_json,
        out_dir=out_dir,
        topk=list(topk),
    )
    agg, df = ev.evaluate()
    ev.write_outputs(agg, df)
    if return_df:
        try:
            from IPython.display import display
            print("=== Per-query metrics (head) ===")
            display(df.head(10))
        except Exception:
            print(df.head(10))
    print("\n=== Summary (Text-Only) ===")
    print(f"TextF1: {agg.get('mean_text_f1', 0.0):.4f}  "
          f"CtxF1: {agg.get('mean_ctx_f1', 0.0):.4f}  "
          f"tR@3: {agg.get('mean_text_recall@3', 0.0):.4f}  "
          f"ChunkOverlap: {agg.get('mean_chunk_overlap', 0.0):.4f}  "
          f"BLEU: {agg.get('mean_bleu', 0.0):.4f}")
    print(f"\nWrote outputs to: {out_dir}\n - per_query_metrics.csv\n - summary.json\n - report.txt")
    return agg, df


if __name__ == "__main__":
    agg, df = evaluate_cs_rag_textonly(
    multi_csv="queries/single_file_retrieval_queries.csv",
    single_csv="queries/single_file_retrieval_queries.csv",
    results_json="results/rag_results.json",
    out_dir="evaluator/evaluation_results"
)
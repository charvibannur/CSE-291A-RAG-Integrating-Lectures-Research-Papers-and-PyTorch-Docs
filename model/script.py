#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_rag_and_eval")


def find_repo_root(script_path: Path) -> Path:
    return (script_path.parent.parent.parent).resolve()


def run_script(python: str, script: str, env: dict | None = None) -> int:
    cmd = [python, script]
    logger.info("Running: %s", " ".join(cmd))
    try:
        # stream output to console
        proc = subprocess.run(cmd, check=True, env=env)
        return proc.returncode
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with exit code %s", e.returncode)
        return e.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAG pipeline then evaluator")
    parser.add_argument("--python", help="Python executable to use (default: env PYTHON or sys.executable)")
    parser.add_argument("--rag-script", default="baselines/baseline2/rag_semanticsearch.py",
                        help="Path to RAG script (relative to repo root)")
    parser.add_argument("--eval-script", default="evaluator/eval.py",
                        help="Path to evaluator script (relative to repo root)")
    parser.add_argument("--openai-key", help="Optional OpenAI API key to set for subprocesses")
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    os.chdir(repo_root)
    logger.info("Repo root: %s", repo_root)

    python = args.python or os.environ.get("PYTHON") or sys.executable
    # If python is something like 'python3 -u' we still check the binary
    python_bin = python.split()[0]
    if shutil.which(python_bin) is None:
        logger.error("Python executable not found: %s", python_bin)
        return 2

    env = os.environ.copy()
    if args.openai_key:
        env["OPENAI_API_KEY"] = args.openai_key
    else:
        if "OPENAI_API_KEY" not in env:
            logger.warning("OPENAI_API_KEY is not set. Embeddings may fail if required.")

    # Resolve script paths relative to repo root
    rag_script = str((repo_root / args.rag_script).resolve())
    eval_script = str((repo_root / args.eval_script).resolve())

    if not Path(rag_script).exists():
        logger.error("RAG script not found: %s", rag_script)
        return 3
    if not Path(eval_script).exists():
        logger.error("Evaluator script not found: %s", eval_script)
        return 4

    rc = run_script(python, rag_script, env=env)
    if rc != 0:
        logger.error("Aborting: RAG script failed (rc=%d)", rc)
        return rc

    rc2 = run_script(python, eval_script, env=env)
    if rc2 != 0:
        logger.error("Evaluator failed (rc=%d)", rc2)
        return rc2

    logger.info("Completed successfully. Retrieval results and evaluation outputs are in the repository paths used by the scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

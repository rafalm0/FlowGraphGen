#!/usr/bin/env python3
"""
Run baseline vs graphflow-regeneration experiments on a local prepared dataset.

Expected dataset structure:
data/
├── metadata.csv
├── <instance_id>/
│   ├── patch.diff                  or gold_patch.diff
│   ├── problem_statement.txt
│   ├── original/
│   │   └── path/to/file.py
│   └── patched/
│       └── path/to/file.py

This script:
1) reads metadata.csv
2) loads the original code, patched code, gold patch, and problem statement
3) asks the model for:
   - baseline patch directly from original code
   - graph representation from original code
   - regenerated code from that graph
   - patch from regenerated code
4) saves JSONL predictions for harness use
5) computes CodeBLEU on FULL FILES, not patches
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional
import ast
from together import Together

import pandas as pd
from groq import Groq
from codebleu import calc_codebleu

# =========================
# CONFIG
# =========================

# MODEL = "llama-3.1-8b-instant"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MAX_CODE_CHARS = 12000
MAX_GRAPH_CHARS = 6000


# =========================
# LLM CLIENT
# =========================

def make_client(api_key: str):
    return Together(api_key=api_key)
    # return Groq(api_key=api_key)


def get_completion(client: Groq, prompt: str, temp: float = 0.2) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return response.choices[0].message.content or ""


# =========================
# IO HELPERS
# =========================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def append_jsonl(path: Path, data: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def find_first_file(root: Path) -> Optional[Path]:
    if not root.exists():
        return None

    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        return None

    if len(files) > 1:
        raise ValueError(f"Expected exactly 1 file under {root}, found {len(files)}")

    return files[0]


def get_patch_path(sample_dir: Path) -> Path:
    candidates = [
        sample_dir / "patch.diff",
        sample_dir / "gold_patch.diff",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No patch file found in {sample_dir}")


# =========================
# PROMPTS
# =========================


def build_graph_prompt(code: str) -> str:
    return f"""
Extract the underlying program structure of the following Python code as a Mermaid graph.

Rules:
- Output ONLY Mermaid
- Start with: graph TD
- No explanations
- Capture the main control flow, important conditions, key helper calls, and major logic steps
- Keep the graph concise and readable
- Use function and method names when helpful
- Do not include commentary outside the graph
- Before sending, remove anything before and after the diff.
- Your output will be immediatly applied as git diff.


PYTHON CODE:
{code[:MAX_CODE_CHARS]}
""".strip()


def build_regen_code_prompt(graph: str, code: str) -> str:
    return f"""
You are an expert Python software engineer specializing in refactoring and code-quality improvements.

Refactor the following Python code to improve its quality while preserving its behavior.

Use the provided structural information about the code to guide the refactoring. Favor improvements that align with the structure, dependencies, and relationships shown there, but do not introduce unnecessary changes.

Refactoring goals:
- improve readability
- improve maintainability
- reduce unnecessary complexity
- reduce duplication when appropriate
- improve modularity only when clearly beneficial
- align the implementation more cleanly with the provided structural relationships

Constraints:
- Preserve existing functionality.
- Preserve public interfaces, class names, function names, method names, and signatures unless a change is strictly necessary.
- Prefer the smallest effective set of changes.
- Avoid unnecessary restructuring, renaming, or redesign.
- Do not introduce new files, dependencies, or external libraries.
- Keep the code stylistically consistent with the original file.
- Do not change user-visible strings, messages, comments, or outputs unless strictly required.
- Use the graph information as guidance, not as a reason to rewrite unrelated parts.
- Ensure the final result is valid Python.
- Keep the output as close as reasonably possible to the original while improving code quality.

Output constraints:
- Return ONLY the final refactored Python code.
- Do not include explanations.
- Do not include markdown code fences.
- Do not include diff markers.
- Before sending, remove anything before and after the diff.
- Your output will be immediatly applied as git diff.

STRUCTURAL INFORMATION:
{graph[:MAX_GRAPH_CHARS]}

ORIGINAL PYTHON CODE:
{code[:MAX_CODE_CHARS]}
""".strip()


def build_regen_diff_prompt(graph: str, code: str, file_name: str) -> str:
    return f"""
You are an expert Python software engineer specializing in refactoring and code-quality improvements.

Produce a minimal unified diff patch that improves the quality of the following Python code while preserving its behavior.

Use the provided structural information about the code to guide the refactoring. Favor improvements that align with the structure, dependencies, and relationships shown there, but do not introduce unnecessary changes.

Refactoring goals:
- improve readability
- improve maintainability
- reduce unnecessary complexity
- reduce duplication when appropriate
- improve modularity only when clearly beneficial
- align the implementation more cleanly with the provided structural relationships

Constraints:
- Preserve existing functionality.
- Preserve public interfaces, class names, function names, method names, and signatures unless a change is strictly necessary.
- Prefer the smallest effective set of changes.
- Avoid unnecessary restructuring, renaming, code movement, or redesign.
- Do not introduce new files, dependencies, or external libraries.
- Keep the code stylistically consistent with the original file.
- Do not change user-visible strings, messages, comments, or outputs unless strictly required.
- Use the graph information as guidance, not as a reason to rewrite unrelated parts.
- The patch must apply cleanly to the original code.
- The resulting code after applying the patch must be valid Python.

Output constraints:
- Return ONLY a valid unified diff patch against the original code.
- Do not include explanations.
- Do not include markdown code fences.
- Do not output the full rewritten file.
- Do not include any text before or after the diff.
- Before sending, remove anything before and after the diff.
- Your output will be immediatly applied as git diff.

File name:
{file_name}

STRUCTURAL INFORMATION:
{graph[:MAX_GRAPH_CHARS]}

ORIGINAL PYTHON CODE:
{code[:MAX_CODE_CHARS]}
""".strip()


def build_rewrite_code_prompt(code: str) -> str:
    return f"""
You are an expert Python software engineer specializing in refactoring and code-quality improvements.

Refactor the following Python code to improve its quality while preserving its behavior.

Refactoring goals:
- improve readability
- improve maintainability
- reduce unnecessary complexity
- reduce duplication when appropriate
- improve modularity only when clearly beneficial

Constraints:
- Preserve existing functionality.
- Preserve public interfaces, class names, function names, method names, and signatures unless a change is strictly necessary.
- Prefer the smallest effective set of changes.
- Avoid unnecessary restructuring, renaming, or redesign.
- Do not introduce new files, dependencies, or external libraries.
- Keep the code stylistically consistent with the original file.
- Do not change user-visible strings, messages, comments, or outputs unless strictly required.
- Ensure the final result is valid Python.
- Keep the output as close as reasonably possible to the original while improving code quality.

Output constraints:
- Return ONLY the final refactored Python code.
- Do not include explanations.
- Do not include markdown code fences.
- Do not include diff markers.
- Before sending, remove anything before and after the diff.
- Your output will be immediatly applied as git diff.

ORIGINAL PYTHON CODE:
{code[:MAX_CODE_CHARS]}
""".strip()


def build_rewrite_diff_prompt(code: str, file_name: str) -> str:
    return f"""
You are an expert Python software engineer specializing in refactoring and code-quality improvements.

Produce a minimal unified diff patch that improves the quality of the following Python code while preserving its behavior.

Refactoring goals:
- improve readability
- improve maintainability
- reduce unnecessary complexity
- reduce duplication when appropriate
- improve modularity only when clearly beneficial

Constraints:
- Preserve existing functionality.
- Preserve public interfaces, class names, function names, method names, and signatures unless a change is strictly necessary.
- Prefer the smallest effective set of changes.
- Avoid unnecessary restructuring, renaming, code movement, or redesign.
- Do not introduce new files, dependencies, or external libraries.
- Keep the code stylistically consistent with the original file.
- Do not change user-visible strings, messages, comments, or outputs unless strictly required.
- The patch must apply cleanly to the original code.
- The resulting code after applying the patch must be valid Python.

Output constraints:
- Return ONLY a valid unified diff patch against the original code.
- Do not include explanations.
- Do not include markdown code fences.
- Do not output the full rewritten file.
- Do not include any text before or after the diff.
- Before sending, remove anything before and after the diff.
- Your output will be immediatly applied as git diff.

File name:
{file_name}

ORIGINAL PYTHON CODE:
{code[:MAX_CODE_CHARS]}
""".strip()


# =========================
# DIFF HANDLING
# =========================

def extract_diff(text: str) -> Optional[str]:
    if not text:
        return None
    if "diff --git" in text:
        return text[text.index("diff --git"):].strip()
    return None


# =========================
# CODE VALIDATION
# =========================


def validate_python_code(code: str) -> dict:
    """
    Validate whether generated Python code is syntactically valid.

    Returns:
        {
            "is_valid": bool,
            "error_type": str,
            "error_message": str,
        }
    """
    try:
        ast.parse(code)
        compile(code, "<generated>", "exec")
        return {
            "is_valid": True,
            "error_type": "",
            "error_message": "",
        }
    except SyntaxError as e:
        return {
            "is_valid": False,
            "error_type": "SyntaxError",
            "error_message": f"{e.msg} (line {e.lineno}, offset {e.offset})",
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


# =========================
# CODEBLEU
# =========================

def compute_codebleu(reference_code: str, predicted_code: str) -> float:
    if not predicted_code:
        return 0.0
    try:
        result = calc_codebleu([reference_code], [predicted_code], lang="python")
        return float(result["codebleu"])
    except Exception:
        return 0.0


# =========================
# DATA LOADING
# =========================

def load_instance_from_row(row: pd.Series) -> dict:
    sample_dir = Path(row["sample_dir"]).resolve()

    problem_path = Path(row["problem_statement_file"]).resolve()
    patch_path = Path(row["patch_file"]).resolve()

    original_dir = Path(row["original_dir"]).resolve()
    patched_dir = Path(row["patched_dir"]).resolve()

    original_file = find_first_file(original_dir)
    patched_file = find_first_file(patched_dir)

    if original_file is None:
        raise FileNotFoundError(f"No original file found under {original_dir}")
    if patched_file is None:
        raise FileNotFoundError(f"No patched file found under {patched_dir}")

    original_code = read_text(original_file)
    patched_code = read_text(patched_file)
    problem_statement = read_text(problem_path)
    gold_patch = read_text(patch_path)

    file_path = str(original_file.relative_to(original_dir))

    return {
        "instance_id": str(row["instance_id"]),
        "repo": str(row.get("repo", "")),
        "base_commit": str(row.get("base_commit", "")),
        "sample_dir": str(sample_dir),
        "file_path": file_path,
        "problem_statement": problem_statement,
        "gold_patch": gold_patch,
        "original_code": original_code,
        "patched_code": patched_code,
    }


# =========================
# MAIN PIPELINE
# =========================

def process_instance(
        client,
        ex: dict,
        pred_base_diff_path: Path,
        pred_graph_diff_path: Path,
) -> Optional[dict]:
    try:
        instance_id = ex["instance_id"]
        file_path = ex["file_path"]
        original_code = ex["original_code"]
        gold_patch = ex["gold_patch"]
        gold_fixed_code = ex["patched_code"]

        print(f"\n🚀 {instance_id}")

        if not original_code.strip():
            print("⚠️ Empty original code")
            return None

        # ========================
        # BASELINE -> CODE
        # ========================
        baseline_fixed_code = get_completion(
            client,
            build_rewrite_code_prompt(original_code),
            temp=0.2,
        )

        baseline_code_valid = validate_python_code(baseline_fixed_code)
        cb_base_code = compute_codebleu(gold_fixed_code, baseline_fixed_code)

        # ========================
        # BASELINE -> DIFF
        # ========================
        baseline_diff = get_completion(
            client,
            build_rewrite_diff_prompt(original_code, file_path),
            temp=0.2,
        )

        baseline_diff = baseline_diff.replace("\\", '/').replace("```diff", "").replace("```", "")

        baseline_diff_record = {
            "instance_id": instance_id,
            "model_patch": baseline_diff,
            "model_name_or_path": MODEL,
        }
        with open(pred_base_diff_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(baseline_diff_record) + "\n")

        # ========================
        # GRAPH EXTRACTION
        # ========================
        graph = get_completion(
            client,
            build_graph_prompt(original_code),
            temp=0.7,
        )

        # ========================
        # GRAPH -> CODE
        # ========================
        graph_fixed_code = get_completion(
            client,
            build_regen_code_prompt(graph, original_code),
            temp=0.2,
        )

        graph_code_valid = validate_python_code(graph_fixed_code)
        cb_graph_code = compute_codebleu(gold_fixed_code, graph_fixed_code)

        # ========================
        # GRAPH -> DIFF
        # ========================
        graph_diff = get_completion(
            client,
            build_regen_diff_prompt(graph, original_code, file_path),
            temp=0.2,
        )

        graph_diff = graph_diff.replace("\\", '/').replace("```diff", "").replace("```", "")

        graph_diff_record = {
            "instance_id": instance_id,
            "model_patch": graph_diff,
            "model_name_or_path": MODEL,
        }
        with open(pred_graph_diff_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(graph_diff_record) + "\n")

        return {
            "instance_id": instance_id,
            "repo": ex["repo"],
            "base_commit": ex["base_commit"],
            "file_path": file_path,

            # Gold info
            "gold_patch_len": len(gold_patch),
            "gold_fixed_code_len": len(gold_fixed_code),

            # Graph info
            "graph_len": len(graph),

            # Baseline code outputs
            "baseline_code": baseline_fixed_code,
            "baseline_code_len": len(baseline_fixed_code),
            "baseline_code_is_valid": baseline_code_valid["is_valid"],
            "baseline_code_error": baseline_code_valid.get("error"),
            "codebleu_base_code": cb_base_code,

            # Baseline diff outputs
            "baseline_diff": baseline_diff,
            "baseline_diff_len": len(baseline_diff),

            # Graph code outputs
            "graph_code": graph_fixed_code,
            "graph_code_len": len(graph_fixed_code),
            "graph_code_is_valid": graph_code_valid["is_valid"],
            "graph_code_error": graph_code_valid.get("error"),
            "codebleu_graph_code": cb_graph_code,

            # Graph diff outputs
            "graph_diff": graph_diff,
            "graph_diff_len": len(graph_diff),
        }

    except Exception as e:
        print(f"❌ Error in {ex.get('instance_id', 'unknown')}: {e}")
        if "429" in str(e):
            print("⏳ Rate limit sleep")
            time.sleep(60)
        return None


def main():
    METADATA_CSV = "data/metadata.csv"
    OUTPUT_DIR = "outputs"
    API_KEY = "KEY"  # or paste your key here
    MAX_INSTANCES = 500  # e.g. 10 for debugging

    api_key = API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Provide API_KEY or set GROQ_API_KEY")

    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_base_path = output_dir / "baseline_predictions.jsonl"
    pred_graph_path = output_dir / "graph_predictions.jsonl"
    results_path = output_dir / "results_codebleu.csv"

    client = make_client(api_key)

    df = pd.read_csv(METADATA_CSV)

    # =========================
    # MAIN LOOP
    # =========================
    results = []

    for i, (_, row) in enumerate(df.iterrows()):
        if MAX_INSTANCES is not None and i >= MAX_INSTANCES:
            break

        ex = load_instance_from_row(row)
        res = process_instance(client, ex, pred_base_path, pred_graph_path)

        if res:
            results.append(res)
            pd.DataFrame(results).to_csv(results_path, index=False)
        time.sleep(60)
    print(f"\n[done] Results saved to: {results_path}")


if __name__ == "__main__":
    main()

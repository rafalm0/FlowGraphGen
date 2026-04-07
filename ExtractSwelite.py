#!/usr/bin/env python3
"""
One-shot disposable repo pipeline.

Reads a $-separated CSV and, for each row:
1. Clones the repo from GitHub into a temporary folder
2. Checks out the base_commit
3. Saves the original files touched by the patch
4. Applies the patch
5. Saves the patched files
6. Saves the patch text
7. Saves the problem statement
8. Writes one row in a final metadata CSV

No repo caching. Each instance gets a fresh clone, then it is deleted.

Required CSV columns:
- repo
- instance_id
- base_commit
- patch
- problem_statement
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def safe_name(text: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:200] if text else "sample"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8", errors="replace")


def copy_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def parse_changed_files_from_patch(patch_text: str) -> List[str]:
    """
    Extract file paths from lines like:
        diff --git a/foo.py b/foo.py
    Uses the b/ side as the target path.
    """
    changed = []
    pattern = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
    for line in patch_text.splitlines():
        m = pattern.match(line.strip())
        if m:
            _, b = m.groups()
            if b != "/dev/null":
                changed.append(b)

    seen = set()
    result = []
    for p in changed:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def clone_repo_fresh(repo_value: str, parent_dir: Path) -> Tuple[Path, str]:
    """
    Clone a GitHub repo fresh into parent_dir.
    Returns (repo_path, repo_url).
    """
    repo_value = repo_value.strip()
    if "/" not in repo_value:
        raise ValueError(f"Invalid repo format '{repo_value}'. Expected owner/name")

    owner, name = repo_value.split("/", 1)
    repo_url = f"https://github.com/{owner}/{name}.git"
    repo_path = parent_dir / f"{owner}__{name}"

    run_cmd(["git", "clone", repo_url, str(repo_path)], check=True)
    return repo_path, repo_url


def checkout_commit(repo_path: Path, commit: str) -> None:
    """
    Fetch full history if needed and checkout target commit detached.
    """
    # Usually clone gets default refs only. This helps when commit is older or not on default branch.
    run_cmd(["git", "fetch", "--all", "--tags"], cwd=repo_path, check=True)

    try:
        run_cmd(["git", "checkout", "--detach", commit], cwd=repo_path, check=True)
    except subprocess.CalledProcessError:
        # One more attempt in case commit is not in fetched refs yet
        run_cmd(["git", "fetch", "--unshallow"], cwd=repo_path, check=False)
        run_cmd(["git", "fetch", "--all", "--tags"], cwd=repo_path, check=True)
        run_cmd(["git", "checkout", "--detach", commit], cwd=repo_path, check=True)


def save_changed_files(
    repo_dir: Path,
    relative_paths: List[str],
    destination_root: Path,
) -> List[str]:
    saved = []
    for rel in relative_paths:
        src = repo_dir / rel
        if src.exists() and src.is_file():
            dst = destination_root / rel
            copy_file(src, dst)
            saved.append(rel)
    return saved


def apply_patch(repo_dir: Path, patch_file: Path) -> Tuple[bool, str]:
    patch_file = patch_file.resolve()
    try:
        result = run_cmd(
            [
                "git",
                "apply",
                "--allow-empty",
                "--recount",
                "--ignore-space-change",
                "--ignore-whitespace",
                str(patch_file),
            ],
            cwd=repo_dir,
            check=True,
        )
        return True, (result.stdout or "") + (result.stderr or "")
    except subprocess.CalledProcessError as e:
        msg = (e.stdout or "") + "\n" + (e.stderr or "")
        return False, msg.strip()


def process_instance(
    row: pd.Series,
    output_dir: Path,
) -> Dict[str, str]:
    repo_value = str(row["repo"])
    instance_id = str(row["instance_id"])
    base_commit = str(row["base_commit"])
    patch_text = str(row["patch"])
    problem_statement = str(row.get("problem_statement", ""))

    changed_files = parse_changed_files_from_patch(patch_text)

    sample_dir = output_dir / safe_name(instance_id)
    original_dir = sample_dir / "original"
    patched_dir = sample_dir / "patched"
    patch_path = sample_dir / "patch.diff"
    statement_path = sample_dir / "problem_statement.txt"
    log_path = sample_dir / "apply_patch.log"

    sample_dir.mkdir(parents=True, exist_ok=True)
    write_text(patch_path, patch_text)
    write_text(statement_path, problem_statement)

    repo_url = ""
    repo_path_str = ""
    patch_status = "unknown"

    with tempfile.TemporaryDirectory(prefix="repo_tmp_") as tmpdir:
        tmp_root = Path(tmpdir)
        try:
            repo_path, repo_url = clone_repo_fresh(repo_value, tmp_root)
            repo_path_str = str(repo_path)

            checkout_commit(repo_path, base_commit)

            original_saved = save_changed_files(repo_path, changed_files, original_dir)

            patch_applied, apply_message = apply_patch(repo_path, patch_path)
            write_text(log_path, apply_message)

            if patch_applied:
                patched_saved = save_changed_files(repo_path, changed_files, patched_dir)
                patch_status = "applied"
            else:
                patched_saved = []
                patch_status = "patch_failed"

            metadata = {
                "instance_id": instance_id,
                "repo": repo_value,
                "repo_url": repo_url,
                "base_commit": base_commit,
                "sample_dir": str(sample_dir.resolve()),
                "patch_file": str(patch_path.resolve()),
                "problem_statement_file": str(statement_path.resolve()),
                "original_dir": str(original_dir.resolve()),
                "patched_dir": str(patched_dir.resolve()),
                "apply_log_file": str(log_path.resolve()),
                "problem_statement": problem_statement,
                "changed_files": json.dumps(changed_files, ensure_ascii=False),
                "original_files_saved": json.dumps(original_saved, ensure_ascii=False),
                "patched_files_saved": json.dumps(patched_saved, ensure_ascii=False),
                "status": patch_status,
            }

            write_text(
                sample_dir / "metadata.json",
                json.dumps(metadata, indent=2, ensure_ascii=False),
            )
            return metadata

        except Exception as e:
            error_metadata = {
                "instance_id": instance_id,
                "repo": repo_value,
                "repo_url": repo_url,
                "base_commit": base_commit,
                "sample_dir": str(sample_dir.resolve()),
                "patch_file": str(patch_path.resolve()),
                "problem_statement_file": str(statement_path.resolve()),
                "original_dir": str(original_dir.resolve()),
                "patched_dir": str(patched_dir.resolve()),
                "apply_log_file": str(log_path.resolve()),
                "problem_statement": problem_statement,
                "changed_files": json.dumps(changed_files, ensure_ascii=False),
                "original_files_saved": json.dumps([], ensure_ascii=False),
                "patched_files_saved": json.dumps([], ensure_ascii=False),
                "status": f"failed: {type(e).__name__}: {e}",
            }

            write_text(log_path, f"ERROR\n{type(e).__name__}: {e}\n")
            write_text(
                sample_dir / "metadata.json",
                json.dumps(error_metadata, indent=2, ensure_ascii=False),
            )
            return error_metadata


def main() -> None:
    input_path = "swe-bench-lite-test-data-overview.csv"
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep=",")

    required_cols = {"repo", "instance_id", "base_commit", "patch", "problem_statement"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    metadata_rows = []

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        instance_id = str(row["instance_id"])
        print(f"[{idx}/{total}] Processing {instance_id}...", flush=True)
        meta = process_instance(row, output_dir)
        metadata_rows.append(meta)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_csv_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)

    print("\nDone.")
    print(f"Metadata CSV: {metadata_csv_path}")
    print(f"Samples saved under: {output_dir}")


if __name__ == "__main__":
    main()
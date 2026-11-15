#!/usr/bin/env python3
"""
Create a source-only GitHub export from the existing MYR_SupplyLens project.

Default behaviour is conservative:
- includes source, tests, documentation, requirements and small static assets;
- excludes datasets, model weights, databases, secrets, logs, backups and caches;
- writes a timestamped export folder and ZIP beside the project directory;
- never modifies or deletes the original project.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg",
    ".json", ".html", ".css", ".js", ".svg", ".ipynb", ".bat", ".sh",
    ".sql", ".xml"
}
STATIC_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".ico"}
ROOT_FILES = {
    ".gitignore", ".gitattributes", "Dockerfile", "Procfile",
    "Makefile", "LICENSE", "NOTICE", "README", "README.md",
    "README_V2_UPGRADE.md", "requirements-colab.txt",
    "requirements-local.txt", "pyproject.toml", "setup.py",
    "setup.cfg", "runtime.txt", "PATCH_MANIFEST.txt"
}
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", "node_modules",
    "data", "artifacts", "models", "backups", "logs", "cache", "tmp",
    "temp", "database", "db", "dist", "build"
}
EXCLUDED_SUFFIXES = {
    ".sqlite", ".sqlite3", ".db", ".h5", ".keras", ".pkl", ".pickle",
    ".joblib", ".pt", ".pth", ".onnx", ".parquet", ".feather",
    ".csv", ".xlsx", ".xls", ".zip", ".tar", ".gz", ".7z", ".log",
    ".pem", ".key", ".p12", ".pfx"
}
MAX_FILE_BYTES = 5 * 1024 * 1024


def is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    if name in {".env", "secrets.toml"} or name.startswith(".env."):
        return True
    sensitive_tokens = ("secret", "credential", "token", "password", "private_key")
    if any(token in name for token in sensitive_tokens):
        return True
    if name.endswith((".pem", ".key", ".p12", ".pfx")):
        return True
    return False


def is_allowed(path: Path, project: Path) -> tuple[bool, str]:
    rel = path.relative_to(project)
    parts = set(rel.parts)

    if any(part in EXCLUDED_DIRS for part in parts):
        return False, "excluded directory"
    if is_sensitive(path):
        return False, "possible secret/credential"
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False, "data/model/archive/runtime file"
    if path.name.lower() == "secrets.toml":
        return False, "Streamlit secrets"
    if path.stat().st_size > MAX_FILE_BYTES:
        return False, "over 5 MB"
    if path.name in ROOT_FILES or path.suffix.lower() in TEXT_SUFFIXES:
        return True, "source/documentation"
    if path.suffix.lower() in STATIC_SUFFIXES:
        # Only publish static images from conventional UI asset directories.
        if any(part.lower() in {"assets", "static", "images", "img"} for part in rel.parts):
            return True, "static UI asset"
        return False, "image outside assets/static/images"
    return False, "unsupported file type"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project = args.project_dir.expanduser().resolve()
    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project.parent / f"{project.name}_GitHub_Export_{stamp}"
    )

    # Prevent output from being created inside the source tree.
    if output_root == project or project in output_root.parents:
        raise SystemExit("Output directory must be outside the original project directory.")

    repo_name = project.name.lower().replace("_", "-")
    repo_root = output_root / repo_name
    if output_root.exists():
        raise SystemExit(
            f"Output already exists: {output_root}\n"
            "Choose a new --output-dir; the exporter will not overwrite it."
        )

    repo_root.mkdir(parents=True)
    included = []
    excluded = []

    for path in sorted(project.rglob("*")):
        if not path.is_file():
            continue

        # Skip the output tree if it is adjacent to the source and already exists.
        try:
            allowed, reason = is_allowed(path, project)
        except (OSError, ValueError):
            excluded.append((str(path), "unreadable"))
            continue

        if not allowed:
            excluded.append((str(path.relative_to(project)), reason))
            continue

        rel = path.relative_to(project)
        destination = repo_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        included.append((str(rel), destination.stat().st_size, digest))

    # Create a conservative .gitignore if the source project has none.
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "data/\\nartifacts/\\nbackups/\\nlogs/\\n.venv/\\n"
            "__pycache__/\\n.ipynb_checkpoints/\\n.env\\n.env.*\\n"
            ".streamlit/secrets.toml\\n*.sqlite*\\n*.db\\n*.h5\\n"
            "*.keras\\n*.pkl\\n*.joblib\\n*.log\\n",
            encoding="utf-8",
        )

    manifest = [
        f"MYR SupplyLens source export",
        f"Source: {project}",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Included files: {len(included)}",
        f"Excluded files: {len(excluded)}",
        "",
        "INCLUDED FILES (path | bytes | sha256)",
    ]
    manifest.extend(f"{p} | {size} | {digest}" for p, size, digest in included)
    manifest.extend(["", "EXCLUDED FILES (path | reason)"])
    manifest.extend(f"{p} | {reason}" for p, reason in excluded)
    (repo_root / "EXPORT_MANIFEST.txt").write_text(
        "\n".join(manifest), encoding="utf-8"
    )

    archive_path = output_root / f"{repo_name}_source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(repo_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_root))

    print("=" * 68)
    print("MYR SUPPLYLENS — GITHUB SOURCE EXPORT")
    print("=" * 68)
    print("Original project:", project)
    print("Export folder:", repo_root)
    print("ZIP archive:", archive_path)
    print("Included files:", len(included))
    print("Excluded files:", len(excluded))
    print("\nIncluded paths:")
    for rel, size, _ in included:
        print(f"  {rel} ({size:,} bytes)")
    print("\nReview EXPORT_MANIFEST.txt before uploading.")
    print("The original Drive project was not modified.")


if __name__ == "__main__":
    main()

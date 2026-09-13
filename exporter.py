"""
exporter.py
-----------
Writes the dataset to one fixed folder on disk.

A browser download lands wherever the browser is configured to put it, which
is outside the app's control -- so when the dataset needs to end up in a
known, stable location every time, the app writes the files itself instead of
handing them to the browser.

Filenames are stable and each export overwrites the last, so the folder always
holds the current dataset rather than an accumulating pile of dated copies.
Nothing here is lost by overwriting: every file is regenerated from the log.

Framework-agnostic like pipeline.py and recognizer.py -- it takes a DataFrame
and a directory, and knows nothing about Streamlit or where the config lives.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

APPROVED_CSV = "digit_dataset_approved.csv"
FULL_CSV = "digit_dataset_full.csv"
MANIFEST = "manifest.json"
IMAGES_SUBDIR = "images"
RAW_SUBDIR = "raw"

DEFAULT_EXPORT_DIR = "exports"


class ExportError(RuntimeError):
    """Raised when the export directory can't be used."""


def resolve_dir(configured: Optional[str] = None) -> Path:
    """Absolute path of the export folder, without creating it."""
    return Path(os.path.expanduser(configured or DEFAULT_EXPORT_DIR)).resolve()


def _write_csv(frame: pd.DataFrame, path: Path) -> int:
    """Write via a temp file + atomic replace, so an interrupted export can't
    leave a half-written CSV where a complete one used to be."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path.stat().st_size


def _copy_images(
    frame: pd.DataFrame, column: str, src_dir: str, dest: Path
) -> int:
    """Copy the PNGs referenced by one column. Missing files are skipped --
    a row whose image was deleted shouldn't fail the whole export."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in frame[column].dropna().astype(str):
        src = Path(src_dir) / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied += 1
    return copied


def write_export(
    df: pd.DataFrame,
    out_dir: Optional[str] = None,
    *,
    approved_status: str = "approved",
    include_images: bool = False,
    images_dir: Optional[str] = None,
    raw_dir: Optional[str] = None,
) -> Dict:
    """Write the dataset into `out_dir` and return a summary of what landed.

    Always writes the approved CSV, the full CSV, and a manifest. With
    `include_images`, also copies the PNGs for the approved rows.
    """
    target = resolve_dir(out_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ExportError(f"Can't create the export folder {target}: {e}") from e
    if not os.access(target, os.W_OK):
        raise ExportError(f"The export folder {target} isn't writable.")

    approved = (
        df[df["status"] == approved_status] if "status" in df.columns else df
    )

    files: List[Dict] = []
    for frame, name in ((approved, APPROVED_CSV), (df, FULL_CSV)):
        size = _write_csv(frame, target / name)
        files.append({"name": name, "rows": len(frame), "bytes": size})

    images_copied = raw_copied = 0
    if include_images and not approved.empty:
        if images_dir and "image_filename" in approved.columns:
            images_copied = _copy_images(
                approved, "image_filename", images_dir, target / IMAGES_SUBDIR
            )
        if raw_dir and "raw_filename" in approved.columns:
            raw_copied = _copy_images(
                approved, "raw_filename", raw_dir, target / RAW_SUBDIR
            )

    manifest = {
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved_samples": int(len(approved)),
        "total_samples": int(len(df)),
        "submissions": int(df["submission_id"].nunique()) if "submission_id" in df.columns else None,
        "per_digit_approved": (
            {str(d): int((approved["label"].astype(int) == d).sum()) for d in range(10)}
            if not approved.empty else {str(d): 0 for d in range(10)}
        ),
        "images_copied": images_copied,
        "raw_images_copied": raw_copied,
        "files": [f["name"] for f in files],
    }
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2))
    files.append({"name": MANIFEST, "rows": None, "bytes": (target / MANIFEST).stat().st_size})

    return {"dir": str(target), "files": files, "manifest": manifest}

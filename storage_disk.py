"""
storage_disk.py
----------------
Local-disk storage backend: every saved digit becomes a PNG on disk plus one
row in a CSV log that already contains the full 784 flattened pixel values,
so `data/dataset.csv` is itself a ready-to-use MNIST-style dataset.

Samples arrive as *submissions*: one contributor drawing all ten digits in a
single sitting. Every row carries the `submission_id` that groups its ten
siblings and a `status` the admin sets -- pending until reviewed, then
approved or rejected. Rejecting is a soft delete: the rows stay on disk so a
decision can be audited or undone, but they are excluded from the approved
export and the gallery.

This is the default backend -- zero setup, works the moment you run the app.
Switch to `storage_cloud.py` (Google Sheets + Drive) once you want the
dataset to persist across restarts on a host with an ephemeral filesystem,
or to be shared by more than one contributor. See SETUP_CLOUD_STORAGE.md.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LOG_PATH = os.path.join(DATA_DIR, "dataset.csv")

PIXEL_COLUMNS = [f"pixel{i}" for i in range(784)]
META_COLUMNS = [
    "timestamp_utc",
    "submission_id",
    "contributor_email",
    "status",
    "label",
    "image_filename",
    "raw_filename",
]
LOG_COLUMNS = META_COLUMNS + PIXEL_COLUMNS

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED)

REQUIRED_DIGITS = tuple(range(10))


class StorageError(RuntimeError):
    """Raised when the backend can't be used (kept for a consistent interface
    with storage_cloud.py, which raises this for missing credentials)."""


def backend_name() -> str:
    return "disk"


def _ensure_log() -> None:
    """Create the log if absent, and migrate a pre-review-workflow one in place.

    A log written before submissions existed has neither `submission_id` nor
    `status` in its header. Appending new-format rows to it would silently
    shift every column, so the header is upgraded first: existing rows are
    backfilled as already-approved (they were collected before review was a
    thing) and given a synthetic per-row submission id.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)
        return

    header = pd.read_csv(LOG_PATH, nrows=0).columns.tolist()
    if header == LOG_COLUMNS:
        return

    old = pd.read_csv(LOG_PATH).copy()  # .copy() de-fragments before we add columns
    if "status" not in old.columns:
        old["status"] = STATUS_APPROVED
    if "submission_id" not in old.columns:
        old["submission_id"] = (
            ("legacy-" + old["timestamp_utc"].astype(str)) if len(old) else pd.Series(dtype=str)
        )
    missing = {c: "" for c in LOG_COLUMNS if c not in old.columns}
    if missing:
        old = pd.concat([old, pd.DataFrame(missing, index=old.index)], axis=1)
    backup = LOG_PATH + ".pre-submissions.bak"
    if len(old) and not os.path.exists(backup):
        os.replace(LOG_PATH, backup)
    old[LOG_COLUMNS].to_csv(LOG_PATH, index=False)


def _row_for(
    label: int,
    final: np.ndarray,
    raw_rgba: np.ndarray,
    contributor_email: str,
    submission_id: str,
    status: str,
) -> Dict:
    """Write this sample's two PNGs and build its log row."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    image_filename = f"{label}_{ts}.png"
    raw_filename = f"{label}_{ts}_raw.png"

    Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="L").save(
        os.path.join(IMAGES_DIR, image_filename)
    )
    Image.fromarray(raw_rgba.astype(np.uint8), mode="RGBA").save(
        os.path.join(RAW_DIR, raw_filename)
    )

    row = {
        "timestamp_utc": ts,
        "submission_id": submission_id,
        "contributor_email": contributor_email or "",
        "status": status,
        "label": int(label),
        "image_filename": image_filename,
        "raw_filename": raw_filename,
    }
    row.update(dict(zip(PIXEL_COLUMNS, np.clip(final, 0, 255).astype(np.uint8).flatten().tolist())))
    return row


def save_submission(
    samples: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    contributor_email: str = "",
) -> str:
    """Persist one complete ten-digit submission and return its submission_id.

    `samples` maps each label to its (final_28x28, raw_rgba) pair. All ten
    digits are required -- a partial set is a programming error here, since
    the Draw portal will not offer to submit until every digit is drawn.
    The ten rows are appended in a single write so a crash can't leave half
    a submission in the log.
    """
    missing = [d for d in REQUIRED_DIGITS if d not in samples]
    if missing:
        raise StorageError(
            "A submission must contain all ten digits; missing: "
            + ", ".join(str(d) for d in missing)
        )

    _ensure_log()
    submission_id = uuid.uuid4().hex[:12]
    rows = [
        _row_for(d, samples[d][0], samples[d][1], contributor_email, submission_id, STATUS_PENDING)
        for d in REQUIRED_DIGITS
    ]
    pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(LOG_PATH, mode="a", header=False, index=False)
    return submission_id


def save_sample(
    label: int,
    final: np.ndarray,
    raw_rgba: np.ndarray,
    contributor_email: str = "",
    submission_id: Optional[str] = None,
    status: str = STATUS_PENDING,
) -> Dict:
    """Append a single sample. Kept for one-off writes and tests; the Draw
    portal goes through save_submission() instead."""
    _ensure_log()
    row = _row_for(
        label, final, raw_rgba, contributor_email,
        submission_id or uuid.uuid4().hex[:12], status,
    )
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(LOG_PATH, mode="a", header=False, index=False)
    return row


def load_all_df() -> pd.DataFrame:
    """The whole log, with the submission columns backfilled.

    A log written before the review workflow existed has neither column. Its
    rows are treated as already-approved rather than pending, so introducing
    review never retroactively hides data someone already collected.
    """
    _ensure_log()
    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    if "status" not in df.columns:
        df["status"] = STATUS_APPROVED
    if "submission_id" not in df.columns:
        df["submission_id"] = "legacy-" + df["timestamp_utc"].astype(str)

    df["status"] = df["status"].fillna(STATUS_PENDING).astype(str)
    df["submission_id"] = df["submission_id"].astype(str)
    return df


def set_submission_status(submission_id: str, status: str) -> int:
    """Set every row of one submission to `status`. Returns rows changed.

    Rewrites the CSV in place via a temporary file and an atomic replace, so
    an interrupted write can't truncate the dataset.
    """
    if status not in STATUSES:
        raise StorageError(f"Unknown status {status!r}; expected one of {STATUSES}.")

    df = load_all_df()
    if df.empty:
        return 0

    mask = df["submission_id"].astype(str) == str(submission_id)
    n = int(mask.sum())
    if n == 0:
        return 0

    df.loc[mask, "status"] = status
    tmp = LOG_PATH + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, LOG_PATH)
    return n


def get_counts(status: Optional[str] = STATUS_APPROVED) -> Dict[str, int]:
    """Per-digit counts. Defaults to approved rows only; pass status=None to
    count every row regardless of review state."""
    df = load_all_df()
    counts = {str(d): 0 for d in range(10)}
    if df.empty:
        return counts
    if status is not None:
        df = df[df["status"] == status]
    if not df.empty:
        for digit, n in df["label"].astype(int).value_counts().items():
            counts[str(digit)] = int(n)
    return counts

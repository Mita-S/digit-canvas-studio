"""
storage_cloud.py
-----------------
Google Sheets + Google Drive storage backend -- the same approach used by
apcbytes/HandwrittenDigitCollectionApp (see DEPLOY.md there), trimmed down to
just what this app needs. Samples arrive as *submissions* -- one contributor
drawing all ten digits -- and each row carries the `submission_id` grouping its
ten siblings plus a `status` the admin sets (pending -> approved / rejected).
Rejecting is a soft delete: the row stays in the Sheet so the decision can be
audited or undone.

Why Sheets *and* Drive: the Sheet holds a flat, MNIST-style CSV-in-the-cloud
log (one row per sample, 784 pixel columns included) that you can open,
filter or export straight into a training script; Drive holds the actual
PNGs in case you want the images themselves later.

Requires three things in Streamlit secrets (see SETUP_CLOUD_STORAGE.md):

    [gcp_service_account]
    ... the full service-account JSON key, as a TOML table ...

    [storage]
    backend = "cloud"
    gsheet_id = "<the destination Google Sheet's file ID>"
    gdrive_folder_id = "<the destination Drive folder's ID>"

The Sheet needs to be shared (Editor access) with the service account's
`client_email`.

`gdrive_folder_id` is OPTIONAL. Google does not give service accounts any
Drive storage quota, so uploading a PNG into a personal My Drive folder
fails with "Service Accounts do not have storage quota" even when the folder
is shared correctly -- the uploaded file would be *owned* by the robot
account, which has nowhere to put it. Shared drives and domain-wide
delegation avoid this, but both are Google Workspace features.

So: leave `gdrive_folder_id` out and the backend logs to the Sheet alone.
That loses nothing essential -- every row already carries all 784 pixel
values, and the Admin gallery rebuilds its thumbnails from those columns
rather than from Drive.
"""

import io
import uuid
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Tuple

import gspread
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build as build_drive_service
from googleapiclient.http import MediaIoBaseUpload
from PIL import Image

WORKSHEET_NAME = "digits"
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

# 1-based column of `status` in the sheet, for targeted cell updates.
_STATUS_COL = META_COLUMNS.index("status") + 1

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class StorageError(RuntimeError):
    """Raised with a human-readable message when cloud storage isn't configured yet."""


def backend_name() -> str:
    return "cloud" if drive_enabled() else "cloud (Sheets only)"


def _secret(*path: str):
    node = st.secrets
    try:
        for key in path:
            node = node[key]
        return node
    except Exception as exc:
        dotted = ".".join(path)
        raise StorageError(
            f"Missing `{dotted}` in Streamlit secrets. Follow SETUP_CLOUD_STORAGE.md, "
            "then restart the app."
        ) from exc


def _optional_secret(*path: str):
    """Like _secret(), but returns None instead of raising when absent."""
    node = st.secrets
    try:
        for key in path:
            node = node[key]
    except Exception:
        return None
    return node or None


def drive_enabled() -> bool:
    """True when a Drive folder is configured for PNG uploads."""
    return _optional_secret("storage", "gdrive_folder_id") is not None


@st.cache_resource(show_spinner=False)
def _credentials() -> Credentials:
    info = _secret("gcp_service_account")
    return Credentials.from_service_account_info(dict(info), scopes=_SCOPES)


def _wrap_google_error(e: Exception) -> "StorageError":
    """Turn a raw Google auth/API failure into an actionable StorageError.

    Without this a bad key surfaces as an unhandled RefreshError and Streamlit
    replaces the whole page with a stack trace -- which says nothing useful and
    leaks the app's internals to every visitor.
    """
    msg = str(e)
    if "Invalid JWT Signature" in msg or "invalid_grant" in msg:
        return StorageError(
            "The Google service-account key was rejected (invalid JWT signature). "
            "The `private_key` in secrets does not match its `private_key_id` — "
            "this happens when only part of the key was replaced during a rotation. "
            "Re-paste the whole [gcp_service_account] section from the downloaded "
            "JSON key file."
        )
    if "PERMISSION_DENIED" in msg or "403" in msg:
        return StorageError(
            "The service account was denied access to the Sheet. Share the Sheet "
            "with its client_email (Editor), and check the Sheets API is enabled."
        )
    if "404" in msg or "not found" in msg.lower():
        return StorageError("Sheet not found — check `gsheet_id` in secrets.")
    return StorageError(f"Google Sheets backend error: {msg[:300]}")


@st.cache_resource(show_spinner=False)
def _worksheet():
    try:
        gc = gspread.authorize(_credentials())
        sh = gc.open_by_key(_secret("storage", "gsheet_id"))
    except StorageError:
        raise
    except Exception as e:
        raise _wrap_google_error(e) from e
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=2000, cols=len(LOG_COLUMNS))
        ws.append_row(LOG_COLUMNS, value_input_option="RAW")
        return ws

    if ws.row_values(1) != LOG_COLUMNS:
        ws.update("A1", [LOG_COLUMNS])
    return ws


@st.cache_resource(show_spinner=False)
def _drive():
    return build_drive_service("drive", "v3", credentials=_credentials(), cache_discovery=False)


def _upload_png(pil_image: Image.Image, filename: str) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=False)
    file = (
        _drive()
        .files()
        .create(
            body={"name": filename, "parents": [_secret("storage", "gdrive_folder_id")]},
            media_body=media,
            fields="id",
        )
        .execute()
    )
    return file["id"]


def _build_row(label, final, raw_rgba, contributor_email, submission_id, status) -> list:
    """Upload this sample's two PNGs to Drive and build its Sheet row."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    image_filename = f"{label}_{ts}.png"
    raw_filename = f"{label}_{ts}_raw.png"

    # Skipped unless a Drive folder is configured -- see the module docstring
    # on why a service account often cannot write to one.
    if drive_enabled():
        _upload_png(Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="L"), image_filename)
        _upload_png(Image.fromarray(raw_rgba.astype(np.uint8), mode="RGBA"), raw_filename)

    pixels = np.clip(final, 0, 255).astype(np.uint8).flatten().tolist()
    return [
        ts, submission_id, contributor_email or "", status,
        int(label), image_filename, raw_filename,
    ] + pixels


def save_submission(
    samples: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    contributor_email: str = "",
) -> str:
    """Upload one complete ten-digit submission and return its submission_id.

    The ten rows go up in a single append_rows call so a network failure
    part-way cannot leave half a submission in the Sheet.
    """
    missing = [d for d in REQUIRED_DIGITS if d not in samples]
    if missing:
        raise StorageError(
            "A submission must contain all ten digits; missing: "
            + ", ".join(str(d) for d in missing)
        )

    submission_id = uuid.uuid4().hex[:12]
    rows = [
        _build_row(d, samples[d][0], samples[d][1], contributor_email, submission_id, STATUS_PENDING)
        for d in REQUIRED_DIGITS
    ]
    _worksheet().append_rows(rows, value_input_option="RAW")
    _read_all_cached.clear()
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
    row = _build_row(
        label, final, raw_rgba, contributor_email,
        submission_id or uuid.uuid4().hex[:12], status,
    )
    _worksheet().append_row(row, value_input_option="RAW")
    _read_all_cached.clear()
    return dict(zip(LOG_COLUMNS, row))


def set_submission_status(submission_id: str, status: str) -> int:
    """Set every row of one submission to `status`. Returns rows changed.

    Only the status cells are rewritten -- the 784 pixel columns are left
    untouched, which keeps a review action one small update instead of
    re-uploading the whole submission.
    """
    if status not in STATUSES:
        raise StorageError(f"Unknown status {status!r}; expected one of {STATUSES}.")

    ws = _worksheet()
    ids = ws.col_values(META_COLUMNS.index("submission_id") + 1)
    # row 1 is the header, so a match at list index i is sheet row i + 1.
    targets = [i + 1 for i, v in enumerate(ids) if i > 0 and str(v) == str(submission_id)]
    if not targets:
        return 0

    ws.batch_update([
        {"range": gspread.utils.rowcol_to_a1(r, _STATUS_COL), "values": [[status]]}
        for r in targets
    ], value_input_option="RAW")
    _read_all_cached.clear()
    return len(targets)


def _log_version() -> int:
    """Cheap change-detector for cache invalidation: how many timestamps exist."""
    try:
        return len(_worksheet().col_values(1))
    except StorageError:
        raise
    except Exception as e:
        raise _wrap_google_error(e) from e


@st.cache_data(show_spinner=False)
def _read_all_cached(_version: int) -> pd.DataFrame:
    values = _worksheet().get_all_values()
    if len(values) <= 1:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.DataFrame(values[1:], columns=values[0])


def load_all_df() -> pd.DataFrame:
    """The whole log, with the submission columns backfilled.

    A Sheet written before the review workflow existed has neither column;
    its rows are treated as already-approved rather than pending, so adding
    review never retroactively hides data someone already collected.
    """
    df = _read_all_cached(_log_version())
    if df.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    if "status" not in df.columns:
        df = df.copy()
        df["status"] = STATUS_APPROVED
    if "submission_id" not in df.columns:
        df = df.copy()
        df["submission_id"] = "legacy-" + df["timestamp_utc"].astype(str)

    df["status"] = df["status"].replace("", STATUS_PENDING).fillna(STATUS_PENDING).astype(str)
    df["submission_id"] = df["submission_id"].astype(str)
    return df


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

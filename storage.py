"""
storage.py
----------
Picks a storage backend at import time and re-exports its functions, so the
rest of the app just does `import storage` and never has to know whether
digits are landing on local disk or in Google Sheets/Drive.

Backend choice, in priority order:
    1. the DIGIT_STORAGE_BACKEND environment variable
    2. st.secrets["storage"]["backend"] in .streamlit/secrets.toml
    3. "disk" (the zero-setup default)

Both storage_disk.py and storage_cloud.py implement the same interface
(save_submission, save_sample, set_submission_status, load_all_df,
get_counts, backend_name) plus a StorageError class and the status
constants, so swapping backends never touches app.py.
"""

import os

import streamlit as st


def _selected_backend() -> str:
    env = os.environ.get("DIGIT_STORAGE_BACKEND")
    if env:
        return env
    try:
        return st.secrets["storage"]["backend"]
    except Exception:
        return "disk"


BACKEND = _selected_backend()

if BACKEND == "cloud":
    from storage_cloud import (  # noqa: F401
        REQUIRED_DIGITS,
        STATUS_APPROVED,
        STATUS_PENDING,
        STATUS_REJECTED,
        STATUSES,
        StorageError,
        backend_name,
        get_counts,
        load_all_df,
        save_sample,
        save_submission,
        set_submission_status,
    )
else:
    from storage_disk import (  # noqa: F401
        IMAGES_DIR,
        RAW_DIR,
        REQUIRED_DIGITS,
        STATUS_APPROVED,
        STATUS_PENDING,
        STATUS_REJECTED,
        STATUSES,
        StorageError,
        backend_name,
        get_counts,
        load_all_df,
        save_sample,
        save_submission,
        set_submission_status,
    )

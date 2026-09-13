"""
Admin portal
============
Review queue plus a dashboard over the whole dataset.

Contributors submit complete ten-digit sets, which land as `pending`. This
page is where an admin looks at all ten at once and decides: **Approve** puts
the set into the approved dataset, **Reject** is a soft delete -- the rows
stay in the log so the decision is auditable and reversible, but they are
excluded from the approved export and the gallery.

app.py only adds this page to the nav for emails listed under [admin] emails
in secrets.toml (see SETUP_GOOGLE_AUTH.md); the check below re-guards in
case someone reaches this URL directly.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

import auth
import exporter
import storage
from style import image_card, inject, page_header, pills
from visuals import image_to_data_uri, render_pixel_grid

inject()

if not auth.is_admin():
    st.error("You don't have access to the admin portal.")
    st.stop()

page_header("🛠️ Admin", "Review incoming submissions and browse the dataset.")

try:
    df = storage.load_all_df()
except storage.StorageError as e:
    st.warning(str(e))
    st.stop()

if df.empty:
    st.info("No submissions yet — nothing to review until someone completes a set in the Draw portal.")
    st.stop()

df["label"] = df["label"].astype(int)
pixel_cols = [c for c in df.columns if c.startswith("pixel")]


def _grid(row) -> np.ndarray:
    return np.array(row[pixel_cols], dtype=float).reshape(28, 28)


def _submission_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per submission: who, when, how many digits, what status."""
    g = frame.groupby("submission_id")
    return pd.DataFrame({
        "contributor": g["contributor_email"].first(),
        "submitted": g["timestamp_utc"].min(),
        "digits": g["label"].nunique(),
        "status": g["status"].first(),
    }).sort_values("submitted", ascending=False)


submissions = _submission_table(df)
pending = submissions[submissions["status"] == storage.STATUS_PENDING]

# --------------------------------------------------------------------------
# Top-line stats
# --------------------------------------------------------------------------

approved_rows = df[df["status"] == storage.STATUS_APPROVED]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Awaiting review", len(pending))
c2.metric("Approved sets", int((submissions["status"] == storage.STATUS_APPROVED).sum()))
c3.metric("Rejected sets", int((submissions["status"] == storage.STATUS_REJECTED).sum()))
c4.metric("Approved samples", len(approved_rows))

st.divider()

# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------

st.subheader("📥 Review queue")

if pending.empty:
    st.success("Nothing pending — every submission has been reviewed.")
else:
    st.caption(
        f"{len(pending)} submission(s) awaiting a decision. Approving adds all ten "
        "digits to the dataset; rejecting hides them from the dataset and the export "
        "but keeps the rows, so the call can be reversed below."
    )

    for sid, meta in pending.iterrows():
        rows = df[df["submission_id"] == sid].sort_values("label")
        with st.container(border=True):
            head, actions = st.columns([3, 1], gap="large")
            with head:
                st.markdown(f"**{meta['contributor'] or '(unknown)'}** · `{sid}`")
                st.caption(f"Submitted {meta['submitted']} · {meta['digits']} of 10 digits")
                if meta["digits"] != 10:
                    st.warning(
                        f"This set has {meta['digits']} distinct digits, not 10 — "
                        "it predates the complete-set rule or was written directly."
                    )

            thumbs = st.columns(10)
            for col, (_, row) in zip(thumbs, rows.iterrows()):
                with col:
                    st.markdown(
                        image_card(
                            image_to_data_uri(render_pixel_grid(_grid(row), cell=3, show_grid=False)),
                            title=str(int(row["label"])),
                            alt=f"submitted digit {int(row['label'])}",
                        ),
                        unsafe_allow_html=True,
                    )

            with actions:
                if st.button("✅ Approve", key=f"ok_{sid}", type="primary", use_container_width=True):
                    n = storage.set_submission_status(sid, storage.STATUS_APPROVED)
                    st.success(f"Approved {n} sample(s).")
                    st.rerun()
                if st.button("🗑️ Reject", key=f"no_{sid}", use_container_width=True):
                    n = storage.set_submission_status(sid, storage.STATUS_REJECTED)
                    st.warning(f"Rejected {n} sample(s).")
                    st.rerun()

st.divider()

# --------------------------------------------------------------------------
# Every submission, and undoing a decision
# --------------------------------------------------------------------------

st.subheader("All submissions")
st.dataframe(submissions, use_container_width=True)

reviewed = submissions[submissions["status"] != storage.STATUS_PENDING]
if not reviewed.empty:
    with st.expander("Change a decision"):
        sid = st.selectbox(
            "Submission",
            options=list(reviewed.index),
            format_func=lambda s: f"{s} · {reviewed.loc[s, 'contributor']} · {reviewed.loc[s, 'status']}",
        )
        new_status = st.radio(
            "Set status to",
            options=[storage.STATUS_APPROVED, storage.STATUS_REJECTED, storage.STATUS_PENDING],
            horizontal=True,
        )
        if st.button("Apply", use_container_width=False):
            n = storage.set_submission_status(sid, new_status)
            st.success(f"{sid} → {new_status} ({n} sample(s)).")
            st.rerun()

st.divider()

# --------------------------------------------------------------------------
# Browse the approved dataset
# --------------------------------------------------------------------------

st.subheader("Browse samples")

f1, f2, f3 = st.columns(3)
with f1:
    status_filter = st.selectbox("Status", options=["approved", "pending", "rejected", "All"])
with f2:
    digit_filter = st.selectbox("Digit", options=["All"] + list(range(10)))
with f3:
    contributor_filter = st.selectbox(
        "Contributor",
        options=["All"] + sorted(df["contributor_email"].dropna().unique().tolist()),
    )

filtered = df.copy()
if status_filter != "All":
    filtered = filtered[filtered["status"] == status_filter]
if digit_filter != "All":
    filtered = filtered[filtered["label"] == int(digit_filter)]
if contributor_filter != "All":
    filtered = filtered[filtered["contributor_email"] == contributor_filter]

total_matching = len(filtered)
filtered = filtered.sort_values("timestamp_utc", ascending=False).head(40)

if filtered.empty:
    st.caption("No samples match this filter.")
else:
    counts = filtered["label"].value_counts().reindex(range(10), fill_value=0)
    st.markdown(pills({str(d): int(counts[d]) for d in range(10)}), unsafe_allow_html=True)
    gallery_cols = st.columns(8)
    for i, (_, row) in enumerate(filtered.iterrows()):
        contributor = str(row.get("contributor_email", "") or "")[:16]
        with gallery_cols[i % 8]:
            st.markdown(
                image_card(
                    image_to_data_uri(render_pixel_grid(_grid(row), cell=4, show_grid=False)),
                    alt=f"digit {int(row['label'])}",
                ),
                unsafe_allow_html=True,
            )
            st.caption(f"{int(row['label'])} · {contributor}" if contributor else str(int(row["label"])))

    st.caption(f"Showing {len(filtered)} of {total_matching} matching sample(s), most recent first (capped at 40).")

st.divider()

# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

st.subheader("Export")

# Where "save to disk" writes. A browser download lands wherever the browser
# decides; this path is the app's own, so the dataset always ends up in the
# same known place. Env var wins so a deploy can override without editing
# secrets.
_configured = os.environ.get("DIGIT_EXPORT_DIR")
if not _configured:
    try:
        _configured = st.secrets["export"]["dir"]
    except Exception:
        _configured = None
export_dir = exporter.resolve_dir(_configured)

st.caption(f"Save location: `{export_dir}`")
st.caption(
    "Every export overwrites the files already there, so this folder always holds "
    "the current dataset. Change it with `[export] dir` in `.streamlit/secrets.toml` "
    "or the `DIGIT_EXPORT_DIR` environment variable."
)

d1, d2 = st.columns([2, 1], gap="large")
with d2:
    # The cloud backend keeps its PNGs in Drive, not on this machine, so there
    # is nothing local to copy there.
    images_available = hasattr(storage, "IMAGES_DIR")
    with_images = st.checkbox(
        "Include PNG images",
        value=False,
        disabled=not images_available,
        help=(
            "Copies the processed and raw PNGs for approved samples into the folder too."
            if images_available
            else "Unavailable on the cloud backend — the PNGs live in Google Drive."
        ),
    )
with d1:
    if st.button("💾 Save export to disk", type="primary", use_container_width=True):
        try:
            res = exporter.write_export(
                df,
                str(export_dir),
                approved_status=storage.STATUS_APPROVED,
                include_images=with_images,
                images_dir=getattr(storage, "IMAGES_DIR", None),
                raw_dir=getattr(storage, "RAW_DIR", None),
            )
            man = res["manifest"]
            st.success(f"Saved to `{res['dir']}`")
            st.dataframe(
                pd.DataFrame(res["files"]).rename(
                    columns={"name": "file", "rows": "rows", "bytes": "size (bytes)"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            if with_images:
                st.caption(
                    f"{man['images_copied']} processed and {man['raw_images_copied']} "
                    "raw PNG(s) copied (approved samples only)."
                )
        except exporter.ExportError as e:
            st.error(str(e))
        except OSError as e:
            st.error(f"Export failed: {e}")

st.write("")
st.caption("Or download through the browser (destination is your browser's setting):")

e1, e2 = st.columns(2)
with e1:
    st.download_button(
        "⬇️ Approved dataset (CSV)",
        data=approved_rows.to_csv(index=False).encode("utf-8"),
        file_name="digit_dataset_approved.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=approved_rows.empty,
        help="Only approved submissions — this is the one to train on.",
    )
with e2:
    st.download_button(
        "⬇️ Everything, all statuses (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="digit_dataset_full.csv",
        mime="text/csv",
        use_container_width=True,
        help="Includes pending and rejected rows, with the status column.",
    )

# Setting up cloud dataset storage (Google Sheets + Drive)

By default the app saves digits to a local `data/` folder — great for
trying it out, but that folder disappears on restart on hosts with an
ephemeral filesystem (like Streamlit Community Cloud), and it isn't shared
between contributors on different machines. Switching to the cloud backend
fixes both: a Google Sheet becomes your always-on, MNIST-style CSV dataset
log, and a Drive folder holds the original PNGs.

This mirrors the approach in
[HandwrittenDigitCollectionApp](https://github.com/apcbytes/HandwrittenDigitCollectionApp/blob/main/DEPLOY.md),
minus the review/approval workflow this app doesn't need.

## 1. Enable the APIs

In the [Google Cloud Console](https://console.cloud.google.com/) for your
project, enable:
- **Google Sheets API**
- **Google Drive API**

## 2. Create a service account

1. **APIs & Services → Credentials → Create Credentials → Service account.**
2. Give it a name (e.g. `digit-canvas-storage`) — the role can be left blank,
   access is granted per-resource in step 4 instead.
3. Open the new service account → **Keys → Add key → Create new key → JSON**.
   This downloads a JSON file — keep it private, never commit it.
4. Note the service account's `client_email` field inside that JSON
   (looks like `digit-canvas-storage@<project-id>.iam.gserviceaccount.com`).

## 3. Create the destination Sheet and Drive folder

1. Create a new Google Sheet (any name) — this becomes the dataset log. Grab
   its **file ID** from the URL: `https://docs.google.com/spreadsheets/d/<THIS_PART>/edit`.
2. **Optional — and usually not worth it.** Create a Google Drive folder for
   the PNGs. Google gives service accounts **no Drive storage quota**, so an
   upload into a personal My Drive folder fails with *"Service Accounts do
   not have storage quota"* even when the folder is shared correctly — the
   file would be owned by the robot account, which has nowhere to put it.
   Shared drives and domain-wide delegation avoid this, but both need Google
   Workspace. Omit `gdrive_folder_id` and the backend logs to the Sheet
   alone; every row already carries all 784 pixel values, and the Admin
   gallery rebuilds thumbnails from those columns rather than from Drive.

   If you do have Workspace and want the PNGs, grab its
   **folder ID** from the URL: `https://drive.google.com/drive/folders/<THIS_PART>`.
3. **Share both** the Sheet and the folder with the service account's
   `client_email` from step 2, with **Editor** access.

## 4. Fill in `.streamlit/secrets.toml`

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` if you
haven't already (never commit this file), and fill in:

```toml
[storage]
backend = "cloud"
gsheet_id = "the Sheet file ID from step 3"
gdrive_folder_id = "the Drive folder ID from step 3"

[gcp_service_account]
# Paste every field from the downloaded JSON key as its own key here.
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "digit-canvas-storage@<project-id>.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```

`private_key` must keep its `\n` line breaks exactly as they appear in the
JSON file (TOML needs them escaped like that inside a quoted string).

## 5. Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The sidebar will show **Storage backend: cloud ☁️**. Save a test digit and
confirm a `digits` tab appears in your Sheet (created automatically on first
save) with a new row, and two PNGs land in your Drive folder.

## Switching back to local storage

Set `backend = "disk"` under `[storage]` (or just delete the `[storage]`
section — `disk` is the default), or set the environment variable
`DIGIT_STORAGE_BACKEND=disk`, which always wins over `secrets.toml`.

## What's in the Sheet

Each row is one saved digit: `timestamp_utc`, `contributor_email`, `label`,
`image_filename`, `raw_filename`, followed by 784 columns `pixel0`…`pixel783`
— the flattened 28×28 array. That means the Sheet (or its CSV export, via
the download button in either portal) is already a ready-to-use MNIST-style
dataset; you don't need to touch the Drive images at all unless you want the
original PNGs specifically.

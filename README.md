# ✍️ Digit Canvas Studio

A small Streamlit app for building your own handwritten-digit dataset: sign
in with Google, draw all ten digits, watch each one get normalized the same
way MNIST training images are (with a live prediction and a view inside the
network), then submit the set for review — with a separate Admin portal that
approves or rejects each submission.

Inspired by the preprocessing pipeline and Google-backed storage in
[HandwrittenDigitCollectionApp](https://github.com/apcbytes/HandwrittenDigitCollectionApp),
rebuilt with a cleaner UI, a simpler two-portal structure, and Streamlit's
native Google sign-in instead of a custom OAuth flow.

## The two portals

- **Draw** — anyone signed in with Google. Draw a digit, watch each
  preprocessing stage (ink extraction → crop → rescale → center-of-mass
  recentering) rendered live, see a live MLP prediction and the network
  firing layer by layer, then add the digit to your set. A set is only
  submittable once **all ten digits 0–9** are captured; submitting sends the
  whole set for review as `pending`.
- **Admin** — only the email addresses listed under `[admin] emails` in
  secrets. A review queue showing each pending submission's ten digits side
  by side, with **Approve** / **Reject** per set; plus per-digit counts, a
  filterable thumbnail gallery, and CSV exports.

## The review workflow

Every row carries a `submission_id` grouping its ten siblings and a `status`:

| status | meaning |
|---|---|
| `pending` | submitted, waiting for an admin decision — not in the approved dataset |
| `approved` | admin kept it; counted, browsable, and in the approved export |
| `rejected` | admin dropped it — a **soft delete**: rows stay in the log so the decision is auditable and reversible from the admin page, but they're excluded from the gallery and the approved export |

Rejecting never destroys data. To hard-delete, filter the rows out of the CSV
yourself.

Both are gated by Google sign-in — see **SETUP_GOOGLE_AUTH.md**.

## The preprocessing pipeline

1. **Ink extraction** — the canvas is flattened to grayscale and inverted so
   the background is 0 and strokes are bright, matching MNIST's convention.
2. **Crop** to the tight bounding box around the ink.
3. **Rescale** so the digit's longer side fits inside a configurable inner
   box (20px by default), aspect ratio preserved, using a LANCZOS filter.
4. **Center geometrically** by dropping the scaled digit in the middle of a
   blank 28×28 canvas.
5. **Re-center by mass** — the step most from-scratch demos skip. The
   digit's center of mass is computed and the image is shifted so that mass
   lands exactly on the canvas center, which is why real MNIST digits look
   consistently centered even for lopsided shapes like "1" or "7".

Every stage is rendered side by side in the UI, and the final 28×28 array
gets a big, zoomable pixel-grid view with optional per-pixel value labels
and a center-of-mass marker.

## Project layout

```
digit-canvas-app/
├── app.py                        # entry point: login gate + page router
├── auth.py                       # Google sign-in + admin allowlist check
├── style.py                      # shared CSS / layout helpers
├── pages/
│   ├── draw.py                   # Draw portal
│   └── admin.py                  # Admin portal
├── pipeline.py                   # crop / scale / recenter math (framework-agnostic)
├── recognizer.py                 # MNIST-trained MLP: predict + layer activations
├── netviz.py                     # animated SVG of the network firing
├── models/mnist_mlp.joblib       # the trained model (committed; retrain with
│                                 #   python recognizer.py --train)
├── visuals.py                    # arrays -> the images shown in the UI
├── storage.py                    # picks disk vs. cloud backend
├── storage_disk.py               # local-disk backend (default, zero setup)
├── storage_cloud.py              # Google Sheets + Drive backend
├── .streamlit/
│   └── secrets.toml.example      # copy to secrets.toml and fill in
├── requirements.txt
├── SETUP_GOOGLE_AUTH.md          # Google sign-in walkthrough
├── SETUP_CLOUD_STORAGE.md        # Google Sheets/Drive walkthrough
└── README.md
```

`pipeline.py` has no Streamlit import at all, so the preprocessing logic can
be reused or unit-tested independently of the UI.

## Run it locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in secrets.toml -- see SETUP_GOOGLE_AUTH.md (required)
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).
You'll land on a "Sign in with Google" screen first.

## Setup guides

- **[SETUP_GOOGLE_AUTH.md](SETUP_GOOGLE_AUTH.md)** — required. Creating the
  OAuth client in Google Cloud Console, generating a cookie secret, and
  configuring who gets Admin access.
- **[SETUP_CLOUD_STORAGE.md](SETUP_CLOUD_STORAGE.md)** — optional. Wiring up
  Google Sheets + Drive so the dataset persists across restarts and is
  shared across contributors, instead of the zero-setup local-disk default.

## Deploy it

- **Streamlit Community Cloud**: push this repo to GitHub, create a new app
  on [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`,
  then paste your `secrets.toml` contents into the app's Settings → Secrets.
  Remember to update `redirect_uri` to your deployed URL in both the secrets
  and the Google Cloud Console OAuth client.
- **Hugging Face Spaces / Docker / any VM**: works the same way as long as
  you can set the equivalent of `secrets.toml` and reach it over HTTPS
  (Google sign-in requires `https://` in production).

## What's in the dataset

Every saved sample becomes one row: `timestamp_utc`, `submission_id`,
`contributor_email`, `status`, `label`, `image_filename`, `raw_filename`,
plus 784 columns `pixel0`…`pixel783` — the flattened 28×28 array. The
**approved** export is the one to train on; the full export includes pending
and rejected rows with their status. the PNGs
(local disk, or Drive with the cloud backend) are there if you want the
original images too.

## Ideas for going further

- Swap the MLP for a small CNN (the `recognizer.py` interface wouldn't change).
- Per-digit review, so an admin can keep part of a set rather than all or nothing.
- A true hard-delete action alongside the soft reject.

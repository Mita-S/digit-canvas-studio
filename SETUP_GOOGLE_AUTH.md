# Setting up Google sign-in

Both portals (Draw and Admin) sit behind Google sign-in, using Streamlit's
native `st.login`/`st.user` (Streamlit ≥ 1.42, requires `Authlib>=1.3.2` —
already in `requirements.txt`). Anyone who signs in can use the **Draw**
portal; only the email addresses you list get the **Admin** portal.

## 1. Create the OAuth client in Google Cloud

1. Open the [Google Auth Platform](https://console.cloud.google.com/auth) for
   the Google Cloud project you want to use (create a new project first if
   you don't have one).
2. Go to **Branding** and fill in the minimum required fields (app name,
   support email). For local development or a Streamlit Community Cloud
   deployment you can use `example.com` as the **Authorized domain**.
3. Go to **Audience**. While your app is in **Testing** status, only emails
   you add under **Test users** can sign in — add your own Google account
   (and anyone else you want testing this) there.
4. Go to **Clients → Create Client**.
   - Application type: **Web application**
   - Name: anything (e.g. `digit-canvas-studio`) — internal only.
   - Under **Authorized redirect URIs**, add your app's URL with the fixed
     path `/oauth2callback` — this path is not configurable, it's built into
     Streamlit:
     - Local dev: `http://localhost:8501/oauth2callback`
     - Streamlit Community Cloud: `https://<your-app-name>.streamlit.app/oauth2callback`
5. Click **Create** and copy the **Client ID** and **Client secret** — you'll
   paste both into `secrets.toml` next.

Leave the app in **Testing** status while you're the only user; switch it to
**Published** (Audience tab) once you want any Google account to be able to
sign in, not just your listed test users.

## 2. Generate a cookie secret

Streamlit needs a random secret to sign its session cookie. Generate one:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output — you'll use it as `cookie_secret` below.

## 3. Fill in `.streamlit/secrets.toml`

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (this
file must **never** be committed — it's already in `.gitignore`) and fill in
the `[auth]` and `[admin]` sections:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"   # match Cloud Console exactly
cookie_secret = "paste the random hex string from step 2"
client_id = "your-client-id.apps.googleusercontent.com"
client_secret = "your-client-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[admin]
# Any signed-in Google account can use the Draw portal; only emails listed
# here get the Admin portal (comma-separated string or a TOML list both work).
emails = "you@example.com, teammate@example.com"
```

`redirect_uri` must exactly match (scheme, host, port, and the
`/oauth2callback` path) one of the **Authorized redirect URIs** you added in
Cloud Console — mismatches are the most common cause of a sign-in failure.

## 4. Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

You should see a "Sign in with Google" screen. After signing in with one of
your test-user accounts, you land on the **Draw** portal; if your email is
also listed under `[admin] emails`, an **Admin** entry appears in the
sidebar navigation too.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub, create the app on
   [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`.
2. In the app's **Settings → Secrets**, paste the full contents of your local
   `secrets.toml` (Google Sheets/Drive sections too, if you're using the
   cloud storage backend — see `SETUP_CLOUD_STORAGE.md`).
3. Update `redirect_uri` to your deployed URL
   (`https://<your-app-name>.streamlit.app/oauth2callback`) in **both**
   places: the secrets you just pasted, and the Authorized redirect URIs
   list in Google Cloud Console.
4. Once you're happy restricting sign-in to your own Google account(s) is no
   longer needed, switch the OAuth consent screen from **Testing** to
   **Published** in the Audience tab so any Google account can sign in.

## Notes

- The identity cookie expires automatically after 30 days of inactivity;
  that's fixed and not configurable.
- Google sign-in only works over `http://localhost` or a real `https://`
  URL — it isn't supported for an app embedded in an iframe.
- The `[admin] emails` allowlist is this app's own access-control layer, not
  a Streamlit or Google feature — anyone signed in who isn't on that list
  simply never sees the Admin page (and is blocked again even if they guess
  its URL, since `pages/admin.py` re-checks `auth.is_admin()`).

"""
Digit Canvas Studio — entry point
==================================
Gates everything behind Google sign-in, then routes to two separate
portals:

  - Draw  (pages/draw.py)  -- anyone signed in with Google can draw digits,
                              watch the MNIST-style preprocessing pipeline
                              run, and save samples into the shared dataset.
  - Admin (pages/admin.py) -- only emails listed under `[admin] emails` in
                              .streamlit/secrets.toml see this page at all;
                              it's a read-only dashboard over everyone's
                              contributions.

Setup:
  - Google sign-in:      SETUP_GOOGLE_AUTH.md
  - Cloud dataset storage: SETUP_CLOUD_STORAGE.md (optional -- the app
                            works with a zero-setup local-disk backend too)

Run with:  streamlit run app.py
"""

import streamlit as st

import auth

st.set_page_config(
    page_title="Digit Canvas Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

auth.require_login()  # stops here with a "Sign in with Google" screen if needed
auth.user_badge()

draw_page = st.Page("pages/draw.py", title="Draw", icon="✍️", default=True)
pages = [draw_page]

if auth.is_admin():
    admin_page = st.Page("pages/admin.py", title="Admin", icon="🛠️")
    pages.append(admin_page)

nav = st.navigation(pages)
nav.run()

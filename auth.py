"""
auth.py
-------
Thin wrapper around Streamlit's native Google sign-in (`st.login` /
`st.user`, Streamlit >= 1.42), plus a simple email-allowlist check that
decides who gets the Admin portal.

Setup: see SETUP_GOOGLE_AUTH.md. Until `[auth]` exists in
.streamlit/secrets.toml, `st.user.is_logged_in` is always False, so
`require_login()` will show the sign-in screen forever — that's expected
before Google OAuth is wired up, not a bug.
"""

import os
from typing import Set

import streamlit as st

# --- Local testing escape hatch --------------------------------------------
# Google's OAuth flow can't be driven by a browser automation script, which
# makes every page behind require_login() untestable end to end. Setting
# DIGIT_DEV_NO_AUTH=1 substitutes a fake signed-in user so a headless browser
# can reach the Draw and Admin portals.
#
# It must come from a real shell environment variable. Streamlit copies every
# top-level secret into os.environ, so reading the env alone would let anyone
# with access to the deploy's secrets box disable sign-in for all visitors --
# _dev_no_auth() therefore refuses the flag when it also appears in secrets.
# Never set it on a public deployment: it disables sign-in for everyone.
DEV_USER_EMAIL = os.environ.get("DIGIT_DEV_EMAIL", "dev@localhost")


def _in_secrets(key: str) -> bool:
    try:
        return key in st.secrets
    except Exception:
        return False


def _dev_no_auth() -> bool:
    """True only when the bypass came from a genuine shell env var."""
    if os.environ.get("DIGIT_DEV_NO_AUTH") != "1":
        return False
    if _in_secrets("DIGIT_DEV_NO_AUTH"):
        # Came from secrets.toml / the Cloud secrets box, not a shell export.
        return False
    return True


def _admin_emails() -> Set[str]:
    try:
        raw = st.secrets["admin"]["emails"]
    except Exception:
        return set()
    if isinstance(raw, str):
        return {e.strip().lower() for e in raw.split(",") if e.strip()}
    return {str(e).strip().lower() for e in raw}


def is_admin() -> bool:
    """True only for a logged-in user whose email is in [admin] emails."""
    if _dev_no_auth():
        return True
    if not st.user.is_logged_in:
        return False
    email = (getattr(st.user, "email", "") or "").lower()
    return bool(email) and email in _admin_emails()


def require_login() -> None:
    """Block the rest of the page behind a Google sign-in screen."""
    if _dev_no_auth():
        st.warning(
            "⚠️ **DIGIT_DEV_NO_AUTH=1** — sign-in is bypassed and you are "
            f"`{DEV_USER_EMAIL}`. Local testing only; never set this on a deploy.",
            icon="⚠️",
        )
        return
    if st.user.is_logged_in:
        return

    st.markdown("### ✍️ Digit Canvas Studio")
    st.write("Sign in with your Google account to draw digits and contribute to the dataset.")

    auth_configured = True
    try:
        st.secrets["auth"]["client_id"]
    except Exception:
        auth_configured = False

    if auth_configured:
        st.button("🔐 Sign in with Google", type="primary", on_click=st.login)
    else:
        st.error(
            "Google sign-in isn't configured yet. Add an `[auth]` section to "
            "`.streamlit/secrets.toml` — step-by-step guide in **SETUP_GOOGLE_AUTH.md**."
        )
    st.stop()


def current_email() -> str:
    """The signed-in user's email -- or the stand-in when auth is bypassed."""
    if _dev_no_auth():
        return DEV_USER_EMAIL
    return (getattr(st.user, "email", "") or "")


# Session keys holding work that belongs to the signed-in user. They are
# dropped on log-out: st.session_state survives the identity change, so an
# unsubmitted digit set would otherwise carry over to whoever signs in next
# and could be submitted under their email.
_USER_STATE_KEYS = ("set_samples", "digit_label", "canvas_key")


def clear_user_state() -> None:
    """Forget the current user's in-progress work."""
    for key in _USER_STATE_KEYS:
        st.session_state.pop(key, None)


def logout() -> None:
    """Log the user out, discarding their in-progress set first."""
    clear_user_state()
    if _dev_no_auth():
        # There is no Google session to end when auth is bypassed; the most
        # honest thing is to reset the working state and say so.
        st.session_state["_dev_logout_notice"] = True
        return
    st.logout()


def user_badge() -> None:
    """Sidebar block: who you are, and the button to stop being them."""
    with st.sidebar:
        st.markdown("---")
        cols = st.columns([1, 3])
        with cols[0]:
            dev = _dev_no_auth()
            picture = None if dev else getattr(st.user, "picture", None)
            if picture:
                st.image(picture, width=36)
            else:
                st.markdown("### 🧪" if dev else "### 👤")
        with cols[1]:
            if _dev_no_auth():
                st.caption(f"{DEV_USER_EMAIL}")
                st.caption("auth bypassed")
            else:
                st.caption(getattr(st.user, "name", None) or current_email() or "Signed in")
                if getattr(st.user, "name", None):
                    st.caption(current_email())

        n_captured = len(st.session_state.get("set_samples", {}))
        if n_captured:
            st.caption(f"⚠️ {n_captured} unsubmitted digit(s) will be discarded.")

        st.button(
            "🚪 Log out",
            on_click=logout,
            use_container_width=True,
            help=(
                "Clears your in-progress set. Sign-in is bypassed "
                "(DIGIT_DEV_NO_AUTH=1), so this only resets the session."
                if _dev_no_auth()
                else "Ends your Google session for this app and clears your in-progress set."
            ),
        )

        if st.session_state.pop("_dev_logout_notice", False):
            st.info("Session reset. Real sign-out needs DIGIT_DEV_NO_AUTH unset.")

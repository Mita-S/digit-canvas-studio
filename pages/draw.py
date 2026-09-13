"""
Draw portal
===========
Draw a digit, then watch it get normalized the same way MNIST training
images are -- cropped to its ink, rescaled, and re-centered by mass -- with
every stage visualized, ending in a big "pixelated" view of the final 28x28
array that can be saved into the shared dataset.

Reachable by anyone signed in with Google (see SETUP_GOOGLE_AUTH.md).
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas

import auth
import netviz
import recognizer
import storage
from pipeline import run_pipeline
from style import image_card, inject, page_header, pills
from visuals import (
    array_to_csv_bytes,
    array_to_png_bytes,
    draw_bbox_overlay,
    image_to_data_uri,
    ink_array_to_image,
    render_activation_grid,
    render_pixel_grid,
)

inject()
page_header(
    "✍️ Draw",
    "Draw a digit, then watch it get cropped, rescaled and re-centered the "
    "same way MNIST training images are — one step at a time.",
)

contributor_email = auth.current_email()


@st.cache_resource(show_spinner=False)
def load_recognizer():
    """Load the trained MLP once per server process, not once per rerun."""
    return recognizer.load_model()

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Controls")

    stroke_width = st.slider("Pen thickness", min_value=6, max_value=30, value=16, step=1)
    canvas_size_px = st.select_slider("Canvas size (px)", options=[224, 280, 336, 392], value=280)

    st.divider()
    st.caption("Preprocessing")
    threshold = st.slider(
        "Ink threshold",
        min_value=0.0, max_value=60.0, value=10.0, step=1.0,
        help="Pixels darker than this are ignored as noise when finding the digit's bounding box.",
    )
    inner_box = st.slider(
        "Inner fit box (px)",
        min_value=14, max_value=24, value=20, step=1,
        help="The digit is rescaled so its longest side fits inside this many pixels, "
             "then dropped onto the 28×28 canvas — the classic MNIST convention is 20.",
    )

    st.divider()
    st.caption("Pixel grid view")
    show_values = st.checkbox("Show pixel values", value=False)
    show_com = st.checkbox("Show center-of-mass marker", value=True)
    cell_size = st.slider("Zoom", min_value=8, max_value=22, value=14, step=1)

    st.divider()
    st.caption("Network layer view")
    layer_cell = st.slider("Neuron size", min_value=5, max_value=18, value=10, step=1)
    animate = st.checkbox("Animate the firing", value=True)

    st.divider()
    if st.button("🗑️ Clear canvas", use_container_width=True):
        st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1
        st.rerun()

    st.divider()
    st.caption(f"Storage backend: **{storage.backend_name()}**" + (" ☁️" if storage.BACKEND == "cloud" else " 💾"))
    st.markdown("#### 📊 Approved dataset")
    try:
        counts = storage.get_counts()  # approved rows only
        total = sum(counts.values())
        st.markdown(pills({str(d): counts[str(d)] for d in range(10)}), unsafe_allow_html=True)
        st.caption(f"{total} approved sample(s). Submissions awaiting review aren't counted here.")
    except storage.StorageError as e:
        st.warning(str(e))

# --------------------------------------------------------------------------
# Layout: canvas on the left, live pipeline stages on the right
# --------------------------------------------------------------------------

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("1. Draw")
    # Clearing: canvas 0.13 only redraws when the `initial_drawing` payload
    # actually changes, so bump a counter into it -- passing a constant
    # {"objects": []} every rerun would be a no-op. The key is rotated too so
    # the widget itself starts fresh.
    clear_count = st.session_state.get("canvas_key", 0)
    canvas_key = f"canvas_{clear_count}"
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=stroke_width,
        stroke_color="#0f172a",
        background_color="#ffffff",
        height=canvas_size_px,
        width=canvas_size_px,
        drawing_mode="freedraw",
        key=canvas_key,
        initial_drawing={"objects": [], "_clear": clear_count},
        return_image_data=True,
    )
    st.caption("Freehand draw a single digit (0–9). The pipeline updates live as you draw.")

with right:
    st.subheader("2. Normalize, step by step")

    have_drawing = canvas_result.image_data is not None
    result = None
    if have_drawing:
        rgba = np.array(canvas_result.image_data, dtype=np.uint8)
        result = run_pipeline(rgba, threshold=threshold, inner=inner_box, canvas_size=28)

    if not have_drawing or result.is_empty:
        st.info("Nothing detected yet — draw a digit on the canvas to see the pipeline run.")
    else:
        pre_shift_com = ndimage.center_of_mass(result.geo_centered)  # (row, col)
        pre_shift_marker = (pre_shift_com[1], pre_shift_com[0])  # -> (x, y)

        stage_cols = st.columns(4)
        stages = [
            ("Ink + box", draw_bbox_overlay(result.ink, result.bbox, display_size=140),
             "Canvas flattened to grayscale ink; box marks the detected digit."),
            ("Cropped", ink_array_to_image(result.cropped, size=140),
             f"Tight bounding box: {result.cropped.shape[1]}×{result.cropped.shape[0]} px."),
            ("Scaled", render_pixel_grid(result.scaled, cell=max(2, 140 // max(result.scaled.shape))),
             f"Longest side fit to {inner_box}px, aspect ratio kept."),
            ("Geo-centered", render_pixel_grid(result.geo_centered, cell=5, com_marker=pre_shift_marker if show_com else None),
             "Dropped in the middle — dot marks its center of mass, still off-center."),
        ]
        for col, (label, img, desc) in zip(stage_cols, stages):
            with col:
                st.markdown(
                    image_card(image_to_data_uri(img), title=label, alt=label),
                    unsafe_allow_html=True,
                )
                st.caption(desc)

# --------------------------------------------------------------------------
# Final pixelated result
# --------------------------------------------------------------------------

st.write("")
st.subheader("3. Final 28×28, re-centered by mass")

if have_drawing and result is not None and not result.is_empty:
    final_col, info_col = st.columns([1.3, 1], gap="large")

    com_marker = (14.0, 14.0) if show_com else None  # by construction, mass now sits at the center

    with final_col:
        grid_img = render_pixel_grid(result.final, cell=cell_size, show_values=show_values, com_marker=com_marker)
        st.markdown(
            image_card(
                image_to_data_uri(grid_img),
                css_class="final-card",
                alt="Final normalized 28x28 digit",
                stretch=False,
            ),
            unsafe_allow_html=True,
        )

    with info_col:
        dx, dy = result.shift
        st.markdown(
            f'<span class="metric-pill">28 × 28 px</span>'
            f'<span class="metric-pill">shift Δx {dx:+.1f}</span>'
            f'<span class="metric-pill">shift Δy {dy:+.1f}</span>',
            unsafe_allow_html=True,
        )
        st.write("")

        # --- live prediction ------------------------------------------------
        try:
            pred, confidence, probs = recognizer.predict(load_recognizer(), result.final)
            unsure = confidence < 0.60
            st.markdown(
                f'<div class="prediction-card{" unsure" if unsure else ""}">'
                f'<p class="pred-label">{"Best guess" if unsure else "Predicted digit"}</p>'
                f'<p class="pred-digit">{pred}</p>'
                f'<p class="pred-conf">{confidence:.1%} confident</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("All ten probabilities"):
                st.bar_chart(
                    pd.DataFrame({"probability": probs}, index=[str(d) for d in range(10)]),
                    height=180,
                )
            st.caption(
                "Predicted by a small MLP trained on MNIST (98.2% test accuracy) — "
                "it reads the normalized 28×28 array above, not your raw strokes. "
                "It's a sanity check, not the label: pick the real one below."
            )
        except recognizer.RecognizerError as e:
            st.info(f"Prediction unavailable — {e}")

        st.write("")
        st.caption(
            "The **shift** shows how far the digit moved to align its center of "
            "mass with the canvas center — the step that keeps lopsided digits "
            "like 1s and 7s from drifting to one side."
        )

        st.download_button(
            "⬇️ Download PNG (28×28)", data=array_to_png_bytes(result.final),
            file_name="digit_28x28.png", mime="image/png", use_container_width=True,
        )
        st.download_button(
            "⬇️ Download pixel values (CSV)", data=array_to_csv_bytes(result.final),
            file_name="digit_28x28.csv", mime="text/csv", use_container_width=True,
        )
else:
    st.caption("Draw something above to see the final normalized digit here.")

# --------------------------------------------------------------------------
# Save to dataset
# --------------------------------------------------------------------------

st.write("")
st.subheader("4. Inside the network, layer by layer")

if have_drawing and result is not None and not result.is_empty:
    try:
        layers = recognizer.layer_activations(load_recognizer(), result.final)
    except recognizer.RecognizerError as e:
        layers = None
        st.info(f"Layer view unavailable — {e}")

    if layers:
        st.caption(
            "The same forward pass that produced the prediction above, one layer "
            "at a time — every value here is what the network actually computed."
        )

        # --- the network firing ---------------------------------------------
        hidden = next((l for l in layers if l.name.startswith("Hidden")), None)
        out_layer = layers[-1]
        if hidden is not None:
            st.markdown(
                netviz.network_svg(
                    load_recognizer(),
                    result.final,
                    hidden.values,
                    out_layer.values,
                    animate=animate,
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                "Pulses travel along the network's strongest actual weights. Edge colour "
                "is the **sign** of the weight — blue pushes toward a digit, red pushes "
                "away — and its brightness is how hard that connection is working. "
                "Violet neurons are shaded and sized by their real activation; the "
                "winning digit is the ringed node with the bold label."
            )
            st.write("")

        layer_cols = st.columns(len(layers), gap="large")

        for col, layer in zip(layer_cols, layers):
            with col:
                heading = f"{layer.name} · {layer.values.size}"
                if layer.grid is not None:
                    heatmap = render_activation_grid(layer.values, layer.grid, cell=layer_cell)
                    # image-rendering:pixelated keeps the neuron cells crisp
                    # when the browser scales the map up to the column width.
                    st.markdown(
                        image_card(
                            image_to_data_uri(heatmap),
                            title=heading,
                            alt=f"{layer.name} activations",
                            pixelated=True,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="stage-card"><h4>{heading}</h4></div>',
                        unsafe_allow_html=True,
                    )
                    # The output layer is only 10 values and they're probabilities --
                    # a labelled bar per digit says more than a 10-cell heatmap.
                    top = int(np.argmax(layer.values))
                    for digit, prob in enumerate(layer.values):
                        mark = "**" if digit == top else ""
                        st.markdown(
                            f"{mark}{digit}{mark} &nbsp;"
                            f'<span style="color:#64748b">{prob:.1%}</span>',
                            unsafe_allow_html=True,
                        )
                        st.progress(float(min(max(prob, 0.0), 1.0)))
                st.caption(layer.detail)
else:
    st.caption("Draw a digit above to watch it propagate through the network.")

st.write("")
st.subheader("5. Build your ten-digit set")

# The set under construction lives in session state: {label: (final, raw_rgba)}.
# Nothing reaches storage until all ten are captured and submitted, so a
# half-finished set can never land in the dataset.
captured = st.session_state.setdefault("set_samples", {})
remaining = [d for d in range(10) if d not in captured]


def _clear_canvas() -> None:
    st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1


st.markdown(
    pills({str(d): ("✓" if d in captured else "—") for d in range(10)}),
    unsafe_allow_html=True,
)
st.progress(len(captured) / 10.0)
st.caption(
    f"**{len(captured)} of 10** captured as `{contributor_email or 'unknown user'}`."
    + (f" Still needed: {', '.join(str(d) for d in remaining)}." if remaining else "")
)

if have_drawing and result is not None and not result.is_empty:
    add_col, label_col = st.columns([1, 2], gap="large")

    with label_col:
        # Default the picker to the next digit still missing, so the common
        # path (draw 0, 1, 2, ... in order) needs no clicking at all.
        default = remaining[0] if remaining else 0
        label = st.radio(
            "Which digit did you draw?",
            options=list(range(10)),
            horizontal=True,
            index=default,
            key="digit_label",
        )
        if label in captured:
            st.caption(f"You already have a **{label}** — adding it again replaces that one.")

    with add_col:
        st.write("")
        verb = "Replace" if label in captured else "Add"
        if st.button(f"➕ {verb} digit {label}", type="primary", use_container_width=True):
            st.session_state["set_samples"][label] = (result.final.copy(), result.raw_rgba.copy())
            _clear_canvas()
            st.rerun()
else:
    st.caption("Draw a digit above, then add it to your set here.")

if captured:
    st.write("")
    st.caption("Your set so far — click a digit to drop it and draw that one again.")
    thumb_cols = st.columns(10)
    for d in range(10):
        with thumb_cols[d]:
            if d in captured:
                st.markdown(
                    image_card(
                        image_to_data_uri(render_pixel_grid(captured[d][0], cell=3, show_grid=False)),
                        title=str(d),
                        alt=f"your digit {d}",
                    ),
                    unsafe_allow_html=True,
                )
                if st.button("✕", key=f"drop_{d}", use_container_width=True, help=f"Remove your {d}"):
                    del st.session_state["set_samples"][d]
                    st.rerun()
            else:
                st.markdown(
                    f'<div class="stage-card" style="opacity:.45"><h4>{d}</h4>'
                    f'<p style="margin:18px 0;color:#94a3b8">not yet</p></div>',
                    unsafe_allow_html=True,
                )

st.write("")
submit_col, reset_col = st.columns([2, 1], gap="large")

with submit_col:
    if remaining:
        st.button(
            f"📤 Submit set for review — {len(remaining)} digit(s) to go",
            disabled=True,
            use_container_width=True,
        )
    elif st.button("📤 Submit all 10 for review", type="primary", use_container_width=True):
        try:
            sid = storage.save_submission(captured, contributor_email=contributor_email)
            st.session_state["set_samples"] = {}
            _clear_canvas()
            st.success(
                f"Submitted as `{sid}` — all 10 digits are now **pending review**. "
                "An admin decides whether the set is kept; it won't appear in the "
                "approved dataset until then."
            )
            st.balloons()
        except storage.StorageError as e:
            st.error(str(e))

with reset_col:
    if captured and st.button("Start over", use_container_width=True):
        st.session_state["set_samples"] = {}
        _clear_canvas()
        st.rerun()

st.divider()
st.caption(
    "Pipeline: canvas → grayscale ink → crop to bounding box → resize into a "
    f"{inner_box}px box (aspect preserved) → center of mass alignment on a 28×28 grid."
)

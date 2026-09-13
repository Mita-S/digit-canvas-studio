"""
visuals.py
----------
Rendering helpers that turn raw numpy arrays into the images shown in the
UI: bounding-box overlays, and the big "pixelated" grid view of the final
28x28 digit (with optional per-cell value labels).

Kept dependency-free beyond Pillow/numpy so the app stays quick to install.
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK_COLOR = (30, 41, 59)       # slate-800 -- the "drawn" pixel color
BG_COLOR = (255, 255, 255)     # white background
GRID_COLOR = (226, 232, 240)   # slate-200 -- subtle grid lines
BOX_COLOR = (99, 102, 241)     # indigo-500 -- bounding-box highlight
COM_COLOR = (244, 63, 94)      # rose-500 -- center-of-mass marker
ACT_COLOR = (79, 70, 229)      # indigo-600 -- neuron activation heatmap

try:
    _FONT = ImageFont.load_default()
except Exception:  # pragma: no cover - extremely unlikely
    _FONT = None


def ink_array_to_image(ink: np.ndarray, size: Optional[int] = None) -> Image.Image:
    """Grayscale ink array (0=background, 255=ink) -> a normal white-bg PNG.

    When `size` is given the image is fit into a size x size square while
    keeping its original aspect ratio (letterboxed on white), rather than
    being stretched -- important since crops are rarely square.
    """
    display = np.clip(255.0 - ink, 0, 255).astype(np.uint8)
    img = Image.fromarray(display, mode="L").convert("RGB")
    if size is None:
        return img

    h, w = ink.shape
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("RGB", (size, size), BG_COLOR)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def draw_bbox_overlay(ink: np.ndarray, bbox: Tuple[int, int, int, int], display_size: int = 240) -> Image.Image:
    """Full ink image with the detected bounding box drawn on top."""
    h, w = ink.shape
    scale = display_size / max(h, w)
    disp_w, disp_h = int(w * scale), int(h * scale)

    img = ink_array_to_image(ink).resize((disp_w, disp_h), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    draw.rectangle(
        [x0 * scale, y0 * scale, x1 * scale - 1, y1 * scale - 1],
        outline=BOX_COLOR,
        width=3,
    )
    return img


def render_pixel_grid(
    arr: np.ndarray,
    cell: int = 10,
    show_grid: bool = True,
    show_values: bool = False,
    com_marker: Optional[Tuple[float, float]] = None,
) -> Image.Image:
    """Blow a small array up into a big grid of colored squares -- the
    "pixelated" view. Each square's shade encodes that pixel's intensity.

    `com_marker`, if given, is an (x, y) point (in *array* coordinates) at
    which to draw a small crosshair -- used to show the center of mass.
    """
    h, w = arr.shape
    img = Image.new("RGB", (w * cell, h * cell), BG_COLOR)
    draw = ImageDraw.Draw(img)

    clipped = np.clip(arr, 0, 255)
    for y in range(h):
        for x in range(w):
            v = clipped[y, x]
            t = v / 255.0
            color = tuple(int(BG_COLOR[i] + (INK_COLOR[i] - BG_COLOR[i]) * t) for i in range(3))
            draw.rectangle([x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1], fill=color)

    if show_grid:
        for x in range(w + 1):
            draw.line([(x * cell, 0), (x * cell, h * cell)], fill=GRID_COLOR)
        for y in range(h + 1):
            draw.line([(0, y * cell), (w * cell, y * cell)], fill=GRID_COLOR)

    if show_values and _FONT is not None and cell >= 16:
        for y in range(h):
            for x in range(w):
                v = int(clipped[y, x])
                text_color = (255, 255, 255) if v > 140 else (100, 116, 139)
                draw.text((x * cell + 2, y * cell + 1), str(v), fill=text_color, font=_FONT)

    if com_marker is not None:
        cx, cy = com_marker
        px, py = cx * cell, cy * cell
        r = max(3, cell // 3)
        draw.line([(px - r, py), (px + r, py)], fill=COM_COLOR, width=2)
        draw.line([(px, py - r), (px, py + r)], fill=COM_COLOR, width=2)
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=COM_COLOR)

    return img


def render_activation_grid(
    values: np.ndarray,
    grid: Tuple[int, int],
    cell: int = 12,
    show_grid: bool = True,
) -> Image.Image:
    """Lay one network layer's activations out as a heatmap.

    Unlike render_pixel_grid, which assumes a fixed 0..255 ink scale, this
    normalizes against the layer's own peak -- hidden-layer activations have
    no fixed range, and a layer whose strongest neuron reads 3.7 should still
    show full contrast rather than near-black.
    """
    rows, cols = grid
    arr = np.asarray(values, dtype=np.float64).reshape(rows, cols)
    peak = float(np.abs(arr).max())
    norm = np.abs(arr) / peak if peak > 0 else np.zeros_like(arr)

    img = Image.new("RGB", (cols * cell, rows * cell), BG_COLOR)
    draw = ImageDraw.Draw(img)
    for y in range(rows):
        for x in range(cols):
            t = norm[y, x]
            color = tuple(int(BG_COLOR[i] + (ACT_COLOR[i] - BG_COLOR[i]) * t) for i in range(3))
            draw.rectangle(
                [x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1], fill=color
            )

    if show_grid and cell >= 6:
        for x in range(cols + 1):
            draw.line([(x * cell, 0), (x * cell, rows * cell)], fill=GRID_COLOR)
        for y in range(rows + 1):
            draw.line([(0, y * cell), (cols * cell, y * cell)], fill=GRID_COLOR)

    return img


def image_to_data_uri(img: Image.Image) -> str:
    """PIL image -> a base64 data: URI, for embedding straight into HTML.

    Streamlit's st.image serves through its /media endpoint, which means a
    broken or blocked fetch renders as the image's alt text instead of the
    picture. Inlining the bytes removes that failure mode entirely -- these
    heatmaps are a few KB each, so the size cost is negligible.
    """
    import base64
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """uint8-clip an array and encode it as PNG bytes (for downloads)."""
    import io

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def array_to_csv_bytes(arr: np.ndarray) -> bytes:
    """Row-major CSV of integer pixel values, for anyone who wants the raw data."""
    import io

    buf = io.StringIO()
    np.savetxt(buf, np.clip(arr, 0, 255).astype(np.uint8), fmt="%d", delimiter=",")
    return buf.getvalue().encode("utf-8")

"""
pipeline.py
-----------
The image-processing core of Digit Canvas Studio.

Implements the classic MNIST-style normalization pipeline used to turn a
freehand digit drawing into a clean 28x28 grayscale array:

    1. Flatten the canvas (RGBA -> single-channel "ink intensity")
    2. Crop to the tight bounding box of the ink
    3. Rescale so the longer side fits inside a 20x20 box (aspect preserved)
    4. Drop the scaled digit onto a blank 28x28 canvas, geometrically centered
    5. Re-center by *center of mass* rather than bounding box -- this is the
       detail that makes digits line up the way real MNIST samples do,
       since a "7" and a "1" have very different visual weight distribution.

Every stage is kept around (not just the final array) so the UI can show
the whole journey from ink to normalized digit.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class PipelineResult:
    """Holds every intermediate array produced while normalizing a digit."""

    raw_rgba: np.ndarray                 # the untouched canvas capture
    ink: Optional[np.ndarray]            # grayscale ink intensity, 0..255
    bbox: Optional[Tuple[int, int, int, int]]  # (x0, y0, x1, y1) of the ink
    cropped: Optional[np.ndarray]        # ink cropped to bbox
    scaled: Optional[np.ndarray]         # cropped, resized to fit a 20x20 box
    geo_centered: Optional[np.ndarray]   # scaled digit dropped on a 28x28 canvas
    final: Optional[np.ndarray]          # geo_centered, re-aligned by mass
    shift: Tuple[float, float]           # (dx, dy) applied during mass-centering
    is_empty: bool                       # True if nothing was drawn


def canvas_to_ink(rgba: np.ndarray) -> np.ndarray:
    """Composite an RGBA canvas onto white, convert to grayscale, then invert
    so background is 0 and dark strokes become high-intensity "ink" -- the
    same convention MNIST uses.
    """
    rgb = rgba[..., :3].astype(np.float64)
    if rgba.shape[-1] == 4:
        alpha = rgba[..., 3:4].astype(np.float64) / 255.0
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)

    gray = np.array(Image.fromarray(rgb.astype(np.uint8)).convert("L"), dtype=np.float64)
    ink = 255.0 - gray
    return np.clip(ink, 0.0, 255.0)


def find_bounding_box(ink: np.ndarray, threshold: float = 10.0):
    """Tight box around every pixel brighter than `threshold`. None if blank."""
    mask = ink > threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def scale_to_inner_box(cropped: np.ndarray, inner: int = 20) -> np.ndarray:
    """Resize so the longer edge equals `inner`, aspect ratio preserved,
    using a high-quality (LANCZOS) filter -- matches how MNIST digits were
    originally normalized from NIST scans.
    """
    h, w = cropped.shape
    scale = inner / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img, dtype=np.float64)


def paste_geometric_center(scaled: np.ndarray, canvas_size: int = 28) -> np.ndarray:
    """Drop the scaled digit in the middle of a blank canvas_size x canvas_size
    field. This is the *naive* centering step -- good enough for a symmetric
    digit like "0", but visibly off for lopsided ones like "7" or "1".
    """
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float64)
    h, w = scaled.shape
    y0 = (canvas_size - h) // 2
    x0 = (canvas_size - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = scaled
    return canvas


def recenter_by_mass(canvas: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Shift the image so its center of mass lands on the canvas center.
    This is the step most from-scratch demos skip, and the reason properly
    preprocessed MNIST digits look so consistently centered.
    """
    total = canvas.sum()
    if total <= 0:
        return canvas.copy(), (0.0, 0.0)

    size = canvas.shape[0]
    cy, cx = ndimage.center_of_mass(canvas)
    shift_y = size / 2.0 - cy
    shift_x = size / 2.0 - cx
    shifted = ndimage.shift(
        canvas, shift=(shift_y, shift_x), order=1, mode="constant", cval=0.0
    )
    shifted = np.clip(shifted, 0, 255)
    return shifted, (shift_x, shift_y)


def run_pipeline(
    rgba: np.ndarray,
    threshold: float = 10.0,
    inner: int = 20,
    canvas_size: int = 28,
) -> PipelineResult:
    """Run the full crop -> scale -> center pipeline and keep every stage."""
    ink = canvas_to_ink(rgba)
    bbox = find_bounding_box(ink, threshold=threshold)

    if bbox is None:
        return PipelineResult(
            raw_rgba=rgba,
            ink=ink,
            bbox=None,
            cropped=None,
            scaled=None,
            geo_centered=None,
            final=None,
            shift=(0.0, 0.0),
            is_empty=True,
        )

    x0, y0, x1, y1 = bbox
    cropped = ink[y0:y1, x0:x1]
    scaled = scale_to_inner_box(cropped, inner=inner)
    geo_centered = paste_geometric_center(scaled, canvas_size=canvas_size)
    final, shift = recenter_by_mass(geo_centered)

    return PipelineResult(
        raw_rgba=rgba,
        ink=ink,
        bbox=bbox,
        cropped=cropped,
        scaled=scaled,
        geo_centered=geo_centered,
        final=final,
        shift=shift,
        is_empty=False,
    )

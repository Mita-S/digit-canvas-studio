"""
netviz.py
---------
An animated SVG of the network actually firing: the pooled input image on
the left, the most-active hidden neurons in the middle, the ten output
digits on the right, and pulses travelling along the connections between
them.

Every position, brightness and edge in the drawing comes from the fitted
model and the real forward pass -- the edges are the strongest weights in
`model.coefs_`, and a neuron's glow is its own activation. Nothing here is
decorative stand-in data.

Framework-agnostic like pipeline.py and recognizer.py: it returns an SVG
string, and the caller decides how to display it.
"""

from __future__ import annotations

import html
from typing import Sequence

import numpy as np

# Geometry of the drawing, in SVG user units.
_W, _H = 900, 430
_IMG_X, _IMG_Y, _IMG_CELL = 24, 75, 40   # pooled 7x7 input grid
_HID_X = 470
_OUT_X = 820
_POOL = 7                                 # input is pooled to _POOL x _POOL

# Palette, assigned by the job each color does rather than by taste, and
# validated with the data-viz palette checker (light surface, --pairs all:
# CVD dE 13.0, normal-vision dE 16.3, contrast all >= 3:1).
#
#   edges    -> DIVERGING (polarity): which way a weight pushes.
#   neurons  -> SEQUENTIAL (magnitude): how hard it fired.
#   input    -> neutral: the input really is a grayscale image, and a fourth
#               hue would compete with the two that carry meaning.
_POS = "#2a78d6"       # blue  -- excitatory weight, pushes toward the digit
_NEG = "#e34948"       # red   -- inhibitory weight, pushes away from it
_ACT = "#4a3aa7"       # violet -- neuron activation ramp
_INK = "#334155"       # neutral ink for the input pixels
_MUTED = "#64748b"     # text


def _pool(final: np.ndarray, n: int = _POOL) -> np.ndarray:
    """Mean-pool the 28x28 input down to n x n so it can be drawn as nodes."""
    k = final.shape[0] // n
    return final[: n * k, : n * k].reshape(n, k, n, k).mean(axis=(1, 3))


def _mix(color: str, t: float) -> str:
    """Blend a hex color toward white by 1-t (t=1 is the full color)."""
    t = float(np.clip(t, 0.0, 1.0))
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(255 + (c - 255) * t) for c in (r, g, b))


def network_svg(
    model,
    final: np.ndarray,
    hidden_activations: np.ndarray,
    probabilities: np.ndarray,
    top_hidden: int = 18,
    edges_per_hidden: int = 3,
    animate: bool = True,
) -> str:
    """Build the animated SVG. Returns a complete <svg> element as a string."""
    pooled = _pool(final)
    pooled_norm = pooled / pooled.max() if pooled.max() > 0 else pooled

    # The hidden neurons worth drawing: the most active ones.
    order = np.argsort(hidden_activations)[::-1][:top_hidden]
    hid_vals = hidden_activations[order]
    hid_max = hid_vals.max() if hid_vals.max() > 0 else 1.0

    W_in = model.coefs_[0]    # (784, n_hidden)
    W_out = model.coefs_[1]   # (n_hidden, 10)

    # Per hidden neuron, aggregate its input weights into the 7x7 pooled
    # blocks so an edge can be drawn to a block rather than to 784 pixels.
    k = final.shape[0] // _POOL
    blocks = W_in.reshape(28, 28, -1)[: _POOL * k, : _POOL * k, :]
    blocks = blocks.reshape(_POOL, k, _POOL, k, -1).sum(axis=(1, 3))  # (7,7,n_hidden)

    hid_y = np.linspace(30, _H - 30, len(order)) if len(order) > 1 else np.array([_H / 2])
    out_y = np.linspace(45, _H - 45, 10)
    winner = int(np.argmax(probabilities))

    parts: list[str] = []

    # --- edges: pooled input block -> hidden neuron -------------------------
    for i, (j, y) in enumerate(zip(order, hid_y)):
        strength = float(hid_vals[i] / hid_max)
        if strength <= 0.02:
            continue
        contrib = blocks[:, :, j] * pooled_norm          # what actually drove it
        flat = np.argsort(np.abs(contrib).ravel())[::-1][:edges_per_hidden]
        for rank, idx in enumerate(flat):
            r, c = divmod(int(idx), _POOL)
            if pooled_norm[r, c] <= 0.05:
                continue
            x0 = _IMG_X + c * _IMG_CELL + _IMG_CELL / 2
            y0 = _IMG_Y + r * _IMG_CELL + _IMG_CELL / 2
            op = 0.14 + 0.60 * strength * (1 - rank / max(1, edges_per_hidden))
            color = _POS if contrib[r, c] >= 0 else _NEG
            cls = "flow" if animate else ""
            parts.append(
                f'<path class="{cls}" d="M{x0:.0f},{y0:.0f} C{(x0 + _HID_X) / 2:.0f},{y0:.0f} '
                f'{(x0 + _HID_X) / 2:.0f},{y:.0f} {_HID_X:.0f},{y:.0f}" '
                f'fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="{op:.2f}" '
                f'style="animation-delay:{0.03 * i:.2f}s" />'
            )

    # --- edges: hidden neuron -> output digit --------------------------------
    for i, (j, y) in enumerate(zip(order, hid_y)):
        strength = float(hid_vals[i] / hid_max)
        if strength <= 0.02:
            continue
        outs = np.argsort(np.abs(W_out[j]))[::-1][:2]
        for o in outs:
            o = int(o)
            positive = W_out[j, o] > 0
            op = 0.12 + 0.55 * strength * (1.0 if o == winner else 0.4)
            color = _POS if positive else _NEG
            cls = "flow" if animate else ""
            parts.append(
                f'<path class="{cls}" d="M{_HID_X:.0f},{y:.0f} C{(_HID_X + _OUT_X) / 2:.0f},{y:.0f} '
                f'{(_HID_X + _OUT_X) / 2:.0f},{out_y[o]:.0f} {_OUT_X:.0f},{out_y[o]:.0f}" '
                f'fill="none" stroke="{color}" stroke-width="1.4" stroke-opacity="{op:.2f}" '
                f'style="animation-delay:{0.4 + 0.03 * i:.2f}s" />'
            )

    # --- input grid ----------------------------------------------------------
    for r in range(_POOL):
        for c in range(_POOL):
            v = float(pooled_norm[r, c])
            x = _IMG_X + c * _IMG_CELL
            y = _IMG_Y + r * _IMG_CELL
            parts.append(
                f'<rect class="{"lit" if animate and v > 0.08 else ""}" '
                f'x="{x}" y="{y}" width="{_IMG_CELL - 3}" height="{_IMG_CELL - 3}" rx="5" '
                f'fill="{_mix(_INK, v)}" stroke="#e2e8f0" stroke-width="1" '
                f'style="animation-delay:{0.02 * (r * _POOL + c):.2f}s" />'
            )

    # --- hidden neurons ------------------------------------------------------
    for i, y in enumerate(hid_y):
        v = float(hid_vals[i] / hid_max)
        # Cap the radius against the row spacing so neighbouring neurons
        # never merge into one solid bar.
        rad = min(3.5 + 5.0 * v, max(4.0, float(np.diff(hid_y).min() if len(hid_y) > 1 else 12) / 2 - 1.5))
        parts.append(
            f'<circle class="{"lit" if animate else ""}" cx="{_HID_X}" cy="{y:.0f}" r="{rad:.1f}" '
            f'fill="{_mix(_ACT, 0.25 + 0.75 * v)}" '
            f'style="animation-delay:{0.4 + 0.03 * i:.2f}s" />'
        )

    # --- output digits -------------------------------------------------------
    for d in range(10):
        p = float(probabilities[d])
        y = out_y[d]
        is_win = d == winner
        rad = 8 + 10 * p
        # The winner is marked by a 2px surface ring and a bold label, not by a
        # hue of its own: status colors are reserved, and "which digit won" is
        # already said in text right beside the node.
        ring = f' stroke="#ffffff" stroke-width="2"' if is_win else ""
        parts.append(
            f'<circle class="{"lit" if animate else ""}" cx="{_OUT_X}" cy="{y:.0f}" r="{rad:.1f}" '
            f'fill="{_mix(_ACT, 0.18 + 0.82 * p)}"{ring} '
            f'style="animation-delay:{0.8 + 0.02 * d:.2f}s" />'
        )
        parts.append(
            f'<text x="{_OUT_X + 34}" y="{y + 4:.0f}" font-size="15" '
            f'font-weight="{700 if is_win else 500}" '
            f'fill="{"#0f172a" if is_win else "#475569"}">{d}</text>'
        )
        parts.append(
            f'<text x="{_OUT_X + 56}" y="{y + 4:.0f}" font-size="12" '
            f'font-weight="{700 if is_win else 400}" '
            f'fill="{"#334155" if is_win else "#94a3b8"}">{p:.0%}</text>'
        )

    labels = [
        (_IMG_X + 3 * _IMG_CELL, 48, "Input · pooled 7×7"),
        (_HID_X, 22, f"Top {len(order)} hidden neurons"),
        (_OUT_X + 10, 22, "Output"),
    ]
    for x, y, text in labels:
        parts.append(
            f'<text x="{x:.0f}" y="{y}" font-size="12" font-weight="700" '
            f'fill="{_MUTED}" text-anchor="middle" letter-spacing="0.5">'
            f"{html.escape(text)}</text>"
        )

    # Edge color encodes the sign of the weight, so it needs a legend --
    # identity is never carried by color alone.
    lx, ly = _IMG_X, _H - 12
    for i, (color, text) in enumerate(
        ((_POS, "excitatory — pushes toward"), (_NEG, "inhibitory — pushes away"))
    ):
        x = lx + i * 210
        parts.append(
            f'<line x1="{x}" y1="{ly - 4}" x2="{x + 22}" y2="{ly - 4}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round" />'
        )
        parts.append(
            f'<text x="{x + 29}" y="{ly}" font-size="11.5" fill="{_MUTED}">'
            f"{html.escape(text)}</text>"
        )

    css = """
      .flow { stroke-dasharray: 5 11; animation: dash 1.15s linear infinite; }
      @keyframes dash { to { stroke-dashoffset: -32; } }
      .lit { animation: glow 1.9s ease-in-out infinite; }
      @keyframes glow { 0%,100% { opacity: .55; } 50% { opacity: 1; } }
      @media (prefers-reduced-motion: reduce) {
        .flow, .lit { animation: none; }
      }
    """ if animate else ""

    return (
        f'<svg viewBox="0 0 {_W} {_H}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Network activity: input pixels through hidden neurons to digit {winner}">'
        f"<style>{css}</style>{''.join(parts)}</svg>"
    )

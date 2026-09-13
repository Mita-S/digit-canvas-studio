"""
recognizer.py
-------------
Live digit prediction for the normalized 28x28 arrays that pipeline.py
produces.

Deliberately framework-agnostic (no Streamlit import), same as pipeline.py,
so the model can be trained from the command line and unit-tested on its
own:

    python recognizer.py --train

That trains a small MLP on MNIST and writes models/mnist_mlp.joblib, which
is what the Draw portal loads at runtime. The model file is small (~1.6 MB)
and is meant to be committed, so a deploy never has to download MNIST or
train anything on the box.

Input convention matters: PipelineResult.final is float 0..255 with the
background at 0 and strokes bright -- exactly MNIST's own convention -- so
the only preparation needed is a /255 scale and a flatten to 784.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "mnist_mlp.joblib"

# The digit classes, in the order predict_proba returns them.
CLASSES = tuple(str(d) for d in range(10))


class RecognizerError(RuntimeError):
    """Raised when the model can't be loaded or a prediction can't be made."""


def prepare(final: np.ndarray) -> np.ndarray:
    """Turn a 28x28 pipeline output into the (1, 784) float row MNIST models
    expect: values scaled to 0..1, row-major flatten.
    """
    arr = np.asarray(final, dtype=np.float64)
    if arr.shape != (28, 28):
        raise RecognizerError(f"expected a 28x28 array, got {arr.shape}")
    return (arr.reshape(1, -1) / 255.0).astype(np.float32)


def load_model(path: Path | str = MODEL_PATH):
    """Load the trained model from disk."""
    import joblib

    path = Path(path)
    if not path.exists():
        raise RecognizerError(
            f"No trained model at {path}. Train one with:  python recognizer.py --train"
        )
    try:
        return joblib.load(path)
    except Exception as e:  # corrupt file, version skew, ...
        raise RecognizerError(f"Could not load the model at {path}: {e}") from e


def predict(model, final: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Predict the digit in a normalized 28x28 array.

    Returns (digit, confidence, probabilities) where `probabilities` is a
    length-10 array indexed by digit and `confidence` is its max.
    """
    x = prepare(final)
    try:
        probs = model.predict_proba(x)[0]
    except Exception as e:
        raise RecognizerError(f"Prediction failed: {e}") from e

    # predict_proba columns follow model.classes_, which are strings ("0".."9")
    # as loaded from OpenML -- remap to a plain 0..9-indexed array so callers
    # never have to care about the label dtype.
    ordered = np.zeros(10, dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        ordered[int(cls)] = probs[col]

    digit = int(np.argmax(ordered))
    return digit, float(ordered[digit]), ordered


# ---------------------------------------------------------------------------
# Layer-by-layer forward pass (for the "inside the network" visualization)
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    """One layer's activations during a single forward pass."""

    name: str
    values: np.ndarray            # 1-D activations for this layer
    grid: Tuple[int, int] | None  # (rows, cols) to display them as, if griddable
    detail: str                   # human-readable note about what happened here


def _grid_shape(n: int) -> Tuple[int, int] | None:
    """Most-square (rows, cols) factorization of n, for laying a vector out as
    a heatmap. None when n is awkwardly prime-ish and a grid would mislead.
    """
    best = None
    for rows in range(int(n**0.5), 0, -1):
        if n % rows == 0:
            best = (rows, n // rows)
            break
    if best is None:
        return None
    rows, cols = best
    # A 1xN "grid" is just the vector again -- not worth drawing as a square.
    return best if rows > 1 and cols / rows <= 8 else None


def _apply(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-z))
    if name == "softmax":
        e = np.exp(z - z.max())
        return e / e.sum()
    return z  # "identity"


def layer_activations(model, final: np.ndarray) -> list[Layer]:
    """Run the forward pass by hand, keeping what every layer produced.

    scikit-learn doesn't expose intermediate activations, so this replays the
    same arithmetic MLPClassifier.predict_proba does -- weights, biases and
    activation functions all read off the fitted model, so it stays correct
    if the architecture is retrained differently.
    """
    if not hasattr(model, "coefs_"):
        raise RecognizerError(
            f"{type(model).__name__} has no coefs_ -- layer activations are "
            "only available for an MLP."
        )

    a = prepare(final)[0]  # (784,), already scaled to 0..1
    layers = [
        Layer(
            name="Input",
            values=a.copy(),
            grid=(28, 28),
            detail=f"{a.size} pixels, scaled to 0–1 — the normalized 28×28 array, flattened.",
        )
    ]

    hidden_act = getattr(model, "activation", "relu")
    out_act = getattr(model, "out_activation_", "softmax")
    n_layers = len(model.coefs_)

    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        is_output = i == n_layers - 1
        z = a @ W + b
        fn = out_act if is_output else hidden_act
        a = _apply(fn, z)

        if is_output:
            name, detail = (
                "Output",
                f"10 class scores through {fn} — these are the probabilities.",
            )
        else:
            live = int((a > 0).sum()) if fn == "relu" else a.size
            name = f"Hidden {i + 1}" if n_layers > 2 else "Hidden"
            detail = (
                f"{a.size} neurons through {fn} — {live} fired "
                f"({live / a.size:.0%}), the rest stayed silent."
            )

        layers.append(
            Layer(
                name=name,
                values=a.copy(),
                grid=None if is_output else _grid_shape(a.size),
                detail=detail,
            )
        )

    return layers


# ---------------------------------------------------------------------------
# Training (offline -- not imported by the app at runtime)
# ---------------------------------------------------------------------------


def train(path: Path | str = MODEL_PATH, verbose: bool = True) -> float:
    """Fetch MNIST, train a small MLP, save it, and return test accuracy."""
    import joblib
    from sklearn.datasets import fetch_openml
    from sklearn.neural_network import MLPClassifier

    # python.org builds on macOS ship without root certificates, so the
    # OpenML download fails with CERTIFICATE_VERIFY_FAILED unless we point
    # OpenSSL at certifi's bundle. Harmless everywhere else.
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    if verbose:
        print("Fetching MNIST (cached after the first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target

    # The canonical MNIST split: first 60k train, last 10k test.
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    if verbose:
        print(f"Training on {X_train.shape[0]} samples...")
    model = MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=40,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=0,
        verbose=verbose,
    )
    model.fit(X_train, y_train)

    accuracy = float(model.score(X_test, y_test))
    if verbose:
        print(f"Test accuracy: {accuracy:.4f}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    if verbose:
        size_mb = path.stat().st_size / 1e6
        print(f"Saved {path} ({size_mb:.1f} MB)")

    return accuracy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the digit recognizer.")
    parser.add_argument("--train", action="store_true", help="train and save the model")
    parser.add_argument("--out", default=str(MODEL_PATH), help="where to write the model")
    args = parser.parse_args()

    if args.train:
        train(args.out)
    else:
        parser.print_help()

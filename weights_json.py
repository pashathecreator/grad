from __future__ import annotations

import json

import numpy as np

from perceptron import MLP


def save_mlp_weights_json(model: MLP, path: str = "mlp_weights.json") -> None:
    layers = []
    for layer in model.layers:
        W = np.asarray(layer.W.data, dtype=np.float32)
        b = np.asarray(layer.b.data, dtype=np.float32).reshape(-1)

        if W.ndim != 2:
            raise ValueError(f"expected W to be 2D, got {W.shape}")
        if b.ndim != 1:
            raise ValueError(f"expected b to be 1D, got {b.shape}")
        if W.shape[1] != b.shape[0]:
            raise ValueError(f"W out dim {W.shape[1]} != b dim {b.shape[0]}")

        layers.append(
            {
                "W_shape": [int(W.shape[0]), int(W.shape[1])],
                "W": W.ravel(order="C").tolist(),
                "b_shape": [int(b.shape[0])],
                "b": b.tolist(),
            }
        )

    dims = {
        "in": int(layers[0]["W_shape"][0]),
        "h1": int(layers[0]["W_shape"][1]),
        "h2": int(layers[1]["W_shape"][1]),
        "out": int(layers[2]["W_shape"][1]),
    }

    payload = {
        "format": "mlp_v1",
        "dtype": "float32",
        "dims": dims,
        "layers": layers,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))

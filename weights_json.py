from __future__ import annotations

import json

from perceptron import MLP


def save_mlp_weights_json(model: MLP, path: str = "mlp_weights.json") -> None:
    layers = []
    for layer in model.layers:
        nin = len(layer.neurons[0].w)
        nout = len(layer.neurons)

        W_flat = []
        for row in range(nin):
            for neuron in layer.neurons:
                W_flat.append(float(neuron.w[row].data))

        b = [float(neuron.b.data) for neuron in layer.neurons]

        if len(W_flat) != nin * nout:
            raise ValueError(f"W size mismatch: {len(W_flat)} != {nin}*{nout}")
        if len(b) != nout:
            raise ValueError(f"b size mismatch: {len(b)} != {nout}")

        layers.append(
            {
                "W_shape": [nin, nout],
                "W": W_flat,
                "b_shape": [nout],
                "b": b,
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

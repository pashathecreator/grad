from __future__ import annotations

from layer import Layer
from value import Value

class MLP:
    def __init__(self) -> None:
        self.layers = [
            Layer(784, 128, activation="relu"),
            Layer(128, 64, activation="relu"),
            Layer(64, 10, activation="linear"),
        ]

    def __call__(self, x: Value) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

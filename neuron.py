from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from value import Value


class Neuron:
    def __init__(self, nin: int, activation: str = "relu") -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: Sequence[Any]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.activation == "relu" else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

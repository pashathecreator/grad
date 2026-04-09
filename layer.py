from __future__ import annotations

import numpy as np
from value import Value

class Layer:
    def __init__(self, nin: int, nout: int, activation: str = 'relu') -> None:
        self.W = Value(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))
        self.b = Value(np.zeros((1, nout)))
        self.activation = activation

    def __call__(self, x: Value) -> Value:
        if x.data.ndim == 1:
            x = x.reshape(1, -1)
        out = x @ self.W + self.b
        if self.activation == 'relu':
            return out.relu()
        return out

    def parameters(self) -> list[Value]:
        return [self.W, self.b]

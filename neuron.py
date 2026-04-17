from math import sqrt
from value import Value
from random import uniform


class Neuron:
    def __init__(self, nin: int, activation: str = "relu") -> None:
        lim = 1.0 / sqrt(nin)
        self.w = [Value(uniform(-lim, lim)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: list[Value]) -> Value:
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return out.relu()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

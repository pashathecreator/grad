from __future__ import annotations
from value import Value
from neuron import Neuron


class Layer:
    def __init__(self, nin: int, nout: int, activation: str = "relu") -> None:
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

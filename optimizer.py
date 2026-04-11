from __future__ import annotations
from value import Value


class Optimizer:
    def __init__(self, parameters: list[Value] | tuple[Value, ...], lr: float = 0.01) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad *= 0.0

    def step(self) -> None:
        for p in self.parameters:
            p.data -= self.lr * p.grad

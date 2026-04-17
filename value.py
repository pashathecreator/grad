from __future__ import annotations
from typing import Any


class Value:
    def __init__(
        self,
        data: Any,
        _parents: tuple[Value, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    def __add__(self, other: Value) -> Value:
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value) -> Value:
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = []
        visited = set()

        def dfs(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    dfs(parent)
                topo.append(v)

        dfs(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

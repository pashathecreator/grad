from __future__ import annotations

import numpy as np

from typing import Any


class Value:
    def __init__(
        self,
        data: Any,
        _parents: tuple[Value, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
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

    def __matmul__(self, other: Value) -> Value:
        out = Value(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(np.maximum(0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sum(self) -> Value:
        out = Value(self.data.sum(), (self,), "sum")

        def _backward() -> None:
            g = float(out.grad)
            self.grad += np.full_like(self.data, g)

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
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

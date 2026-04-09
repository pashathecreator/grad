import numpy as np


def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad

class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def transpose(self):
        out = Value(self.data.T, (self,), 'T')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Value(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Value(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += unbroadcast((-(self.data) / (other.data ** 2)) * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Value(other) / self

    def backward(self):
        topo = []
        visited = set()

        def dfs(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    dfs(parent)
                topo.append(v)

        dfs(self)
        if self.data.size != 1:
            raise RuntimeError("backward() can only be called on a scalar Value")
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

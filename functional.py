from __future__ import annotations

import numpy as np
from value import Value

class Functional:
    @staticmethod
    def softmax_cross_entropy(logits: Value, target: int, eps: float = 1e-12) -> Value:
        x = logits.data
        x = x - x.max(axis=1, keepdims=True)
        exp = np.exp(x)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss_data = -np.log(probs[0, int(target)] + eps)

        out = Value(loss_data, (logits,), 'softmax_ce')

        def _backward() -> None:
            dlogits = probs
            dlogits = dlogits.copy()
            dlogits[0, int(target)] -= 1.0
            logits.grad += out.grad * dlogits

        out._backward = _backward
        return out

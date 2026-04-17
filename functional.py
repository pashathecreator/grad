from __future__ import annotations

import math
from value import Value


class Functional:
    @staticmethod
    def softmax_cross_entropy(
        logits: list[Value], target: int, eps: float = 1e-12
    ) -> Value:
        max_val = max(v.data for v in logits)
        exps = [math.exp(v.data - max_val) for v in logits]
        s = sum(exps)
        probs = [e / s for e in exps]

        loss_data = -math.log(probs[target] + eps)
        out = Value(loss_data, tuple(logits), "softmax_ce")

        def _backward() -> None:
            for i, v in enumerate(logits):
                v.grad += out.grad * (probs[i] - (1.0 if i == target else 0.0))

        out._backward = _backward
        return out

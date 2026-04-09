import random
from value import Value


class Neuron:
    def __init__(self, nin, activation="relu"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.activation == "relu" else act

    def parameters(self):
        return self.w + [self.b]

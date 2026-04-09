from layer import Layer

class MLP:
    def __init__(self):
        self.layers = [
            Layer(784, 128, activation="relu"),
            Layer(128, 64, activation="relu"),
            Layer(64, 10, activation="linear"),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

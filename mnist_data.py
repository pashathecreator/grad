import hashlib
import os
import urllib.request

import numpy as np


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_mnist(path: str = "mnist.npz", verify_hash: bool = True) -> str:
    if os.path.exists(path):
        if verify_hash and _sha256(path) != MNIST_SHA256:
            os.remove(path)
        else:
            return path

    urllib.request.urlretrieve(MNIST_URL, path)
    if verify_hash and _sha256(path) != MNIST_SHA256:
        os.remove(path)
        raise RuntimeError("mnist.npz hash mismatch")
    return path


def load_mnist(path: str = "mnist.npz", normalize: bool = True):
    path = download_mnist(path)
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    if normalize:
        x_train = x_train.astype(np.float64) / 255.0
        x_test = x_test.astype(np.float64) / 255.0

    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))


def sample_to_vector(x28: np.ndarray) -> np.ndarray:
    return x28.reshape(28 * 28)

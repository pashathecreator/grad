from __future__ import annotations

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


def load_mnist(
    path: str = "mnist.npz",
    normalize: bool = True,
) -> tuple[
    tuple[list[list[float]], list[int]],
    tuple[list[list[float]], list[int]],
]:
    path = download_mnist(path)
    with np.load(path, allow_pickle=True) as f:
        x_train_np = f["x_train"]
        y_train_np = f["y_train"]
        x_test_np = f["x_test"]
        y_test_np = f["y_test"]

    def to_flat_floats(arr: np.ndarray) -> list[list[float]]:
        """(N, 28, 28) → list of N flat float lists of length 784"""
        if normalize:
            arr = arr.astype(np.float64) / 255.0
        return [row.reshape(784).tolist() for row in arr]

    x_train = to_flat_floats(x_train_np)
    x_test = to_flat_floats(x_test_np)
    y_train = y_train_np.tolist()
    y_test = y_test_np.tolist()

    return (x_train, y_train), (x_test, y_test)

import numpy as np
from typing import Tuple


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss function and gradient."""
    # YOUR CODE HERE
    loss = np.mean(np.square(y_pred - y_true))
    grad = np.square(y_pred - y_true)
    return loss, grad


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean absolute error loss function and gradient."""
    # YOUR CODE HERE
    loss = np.mean(np.abs(y_pred - y_true))
    grad = np.sign(y_pred - y_true)
    return loss, grad

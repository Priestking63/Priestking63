import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals"""
    return y_true - y_pred


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared errors"""
    ...


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    ...


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE terms"""
    ...


def quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01
) -> np.ndarray:
    """Quantile loss terms"""
    ...
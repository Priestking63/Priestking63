import numpy as np
from typing import List, Tuple, Optional
from scipy import stats
from scipy.stats import shapiro, ttest_1samp, levene, bartlett, fligner
from sklearn.metrics import log_loss

def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals"""
    return y_true - y_pred


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared errors"""
    return np.square(y_true - y_pred)


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any((y_pred < 0) | (y_pred > 1)):
        raise ValueError("y_pred must be between 0 and 1")
    if np.any((y_pred == 0) | (y_pred == 1)):
        raise ValueError("y_pred must be not 0 and 1")
 
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE terms"""
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values")
    
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Input contains negative values")
    
    return 1 - y_pred / y_true


def quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01
) -> np.ndarray:
    """Quantile loss terms"""
    return np.where(y_true > y_pred, q * (y_true - y_pred), (1 - q) * (y_pred - y_true))


def test_normality(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05
) -> Tuple[float, bool]:
    """Normality test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the normality test

    is_rejected : bool
        True if the normality hypothesis is rejected, False otherwise

    """
    _ , p_value = shapiro(y_true-y_pred)
    return float(p_value), bool(p_value < alpha)


def test_unbiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Unbiasedness test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    prefer : str, optional (default=None)
        If None or "two-sided", test whether the residuals are unbiased.
        If "positive", test whether the residuals are unbiased or positive.
        If "negative", test whether the residuals are unbiased or negative.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the unbiasedness hypothesis is rejected, False otherwise

    """
    if prefer in [None, "two-sided"]:
        alternative = "two-sided"
    elif prefer == "positive":
        alternative = "greater"
    elif prefer == "negative":
        alternative = "less"
    else:
        raise ValueError("prefer must be None, 'two-sided', 'positive', or 'negative'")

    _, p_value = ttest_1samp(y_true - y_pred, popmean= 0, alternative=alternative)
    return float(p_value), bool(p_value < alpha)


def test_homoscedasticity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Homoscedasticity test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    bins : int, optional (default=10)
        Number of bins to use for the test.
        All bins are equal-width and have the same number of samples, except
        the last bin, which will include the remainder of the samples
        if n_samples is not divisible by bins parameter.

    test : str, optional (default=None)
        If None or "bartlett", perform Bartlett's test for equal variances.
        If "levene", perform Levene's test.
        If "fligner", perform Fligner-Killeen's test.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the homoscedasticity hypothesis is rejected, False otherwise

    """
    res = y_true - y_pred

    indices = np.argsort(y_pred)
    bin_indices = np.array_split(indices, bins)
    residual_groups = [res[indices] for indices in bin_indices]

    if test in [None, "bartlett"]:
        test_h = bartlett
    elif test == 'levene':
        test_h = levene
    elif test == "fligner":
        test_h = fligner
    else:
        raise ValueError("НЕТ")

    _, p_value = test_h(*residual_groups)
    return float(p_value), bool(p_value < alpha)


def xy_fitted_residuals(y_true, y_pred):
    """Coordinates (x, y) for fitted residuals against true values."""
    residuals = y_true- y_pred
    return y_pred, residuals


def xy_normal_qq(y_true, y_pred):
    """Coordinates (x, y) for normal Q-Q plot."""
    residuals = y_true - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    sample = np.sort(standardized_residuals)
    n = len(residuals)
    probs = np.linspace(0, 1, n, endpoint=False)
    theoretical = stats.norm.ppf(probs)
    sample = np.sort(standardized_residuals)
    return theoretical, sample


def xy_scale_location(y_true, y_pred):
    """Coordinates (x, y) for scale-location plot."""
    residuals = y_true - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))

    return y_pred, sqrt_abs_std_residuals


def fairness(residuals: np.ndarray) -> float:
    """Compute Gini fairness of array of values"""
    abs_errors = np.abs(residuals)
    n = len(abs_errors)

    if n == 0 or np.all(abs_errors == 0):
        return 1.0

    avg_errors = np.mean(abs_errors)
    double_sum = np.sum(np.abs(abs_errors[:, None] - abs_errors[None, :]))

    gini = double_sum / (2 * n**2 * avg_errors)
    return float(1 - gini)


def best_prediction(
    y_true: np.ndarray, y_preds: List[np.ndarray], fairness_drop: float = 0.05
) -> int:
    """Find index of best model"""
    baseline = fairness(logloss(y_true,y_preds[0]))
    base_loss = log_loss(y_true, y_preds[0])
    best_model = 0

    for i in range(1, len(y_preds)):
        cur_fairness = fairness(logloss(y_true, y_preds[i]))
        cur_loss = log_loss(y_true, y_preds[i])

        if (cur_fairness >= baseline * (1 - fairness_drop) and cur_loss < base_loss):
            best_model = i
            base_loss = cur_loss
    return best_model

from typing import Optional

import numpy as np
import pandas as pd
import residuals
from sklearn.metrics import roc_auc_score
from sklearn.base import ClassifierMixin

def best_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k best cases according to the given function"""
    if func is None:
        resid_values = y_test - y_pred
    else:
        residual_func = getattr(residuals, func)
        resid_values = residual_func(y_test, y_pred)

    abs_resid = np.abs(resid_values)
    if mask is not None:
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]
        abs_resid = abs_resid[mask]

    sorted_indices = abs_resid.argsort()

    top_indices = sorted_indices[:top_k]

    result = {
        "X_test": X_test.iloc[top_indices],
        "y_test": y_test.iloc[top_indices],
        "y_pred": y_pred.iloc[top_indices],
        "resid": abs_resid.iloc[top_indices],
    }
    return result


def worst_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k worst cases according to the given function"""
    if func is None:
        resid_values = y_test - y_pred
    else:
        residual_func = getattr(residuals, func)
        resid_values = residual_func(y_test, y_pred)

    abs_resid = np.abs(resid_values)
    if mask is not None:
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]
        abs_resid = abs_resid[mask]

    sorted_indices = abs_resid.argsort()[::-1]

    top_indices = sorted_indices[:top_k]

    result = {
        "X_test": X_test.iloc[top_indices],
        "y_test": y_test.iloc[top_indices],
        "y_pred": y_pred.iloc[top_indices],
        "resid": abs_resid.iloc[top_indices],
    }
    return result


def adversarial_validation(
    classifier: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    quantile: float = 0.1,
    func: Optional[str] = None,
) -> dict:
    """Adversarial validation residual analysis"""
    if func is None:
        resid_values = y_test - y_pred
    else:
        residual_func = getattr(residuals, func)
        resid_values = residual_func(y_test, y_pred)

    abs_resid = np.abs(resid_values)
    threshold = np.quantile(abs_resid, 1 - quantile)
    labels = (abs_resid >= threshold).astype(int)
    classifier.fit(X_test, labels)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(labels, y_prob)
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = dict(zip(X_test.columns, classifier.feature_importances_))
    elif hasattr(classifier, 'coef_'):
        feature_importances = dict(zip(X_test.columns, np.abs(classifier.coef_[0])))
    else:
        raise ValueError("Classifier does not have feature_importances_ or coef_")
    
    return {
        "ROC-AUC": roc_auc,
        "feature_importances": feature_importances,
    }

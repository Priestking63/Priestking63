from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Node:
    """Decision tree node."""

    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""

    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mean squared error of a vector."""
        mse = np.mean((y - np.mean(y)) ** 2)
        return mse

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mean squared error of two vectors."""
        n_left = y_left.shape[0]
        n_right = y_right.shape[0]
        weighted_mse = (n_left * self._mse(y_left) + n_right * self._mse(y_right)) / (
            n_left + n_right
        )
        return weighted_mse

    def _split(self, X: np.ndarray, y: np.ndarray, feature: int) -> np.float64:
        """Find the best split for a node (one feature)"""
        feature_values = X[:, feature]
        best_mse = float("inf")
        best_threshold = None
        sorted_indices = np.argsort(feature_values)
        sorted_features = feature_values[sorted_indices]
        sorted_y = y[sorted_indices]
        unique_vals = np.unique(sorted_features)
        for i in range(len(unique_vals) - 1):
            threshold = unique_vals[i]
            split_index = np.searchsorted(sorted_features, threshold, side="right")
            y_left = sorted_y[:split_index]
            y_right = sorted_y[split_index:]
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            current_mse = self._weighted_mse(y_left, y_right)
            if current_mse < best_mse:
                best_mse = current_mse
                best_threshold = threshold

        return best_threshold, best_mse

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node (all feature)"""
        best_mse = float("inf")
        best_idx = None
        best_thr = None
        n_features = X.shape[1]
        for feature in range(n_features):
            threshold, mse = self._split(X, y, feature)
            if mse < best_mse:
                best_mse = mse
                best_idx = feature
                best_thr = threshold

        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        if (depth >= self.max_depth) or (X.shape[0] < self.min_samples_split):
            return Node(
                n_samples=X.shape[0], value=int(np.round(np.mean(y))), mse=self._mse(y)
            )
        feature, threshold = self._best_split(X, y)
        if threshold is None:
            return Node(
                n_samples=X.shape[0], value=int(np.round(np.mean(y))), mse=self._mse(y)
            )

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        if not (np.any(left_indices) and np.any(right_indices)):
            return Node(
                n_samples=X.shape[0], value=int(np.round(np.mean(y))), mse=self._mse(y)
            )
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        left_node = self._split_node(X_left, y_left, depth + 1)
        right_node = self._split_node(X_right, y_right, depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            n_samples=X.shape[0],
            value=int(np.round(np.mean(y))),
            mse=self._mse(y),
            left=left_node,
            right=right_node,
        )

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        tree_dict = self._as_json(self.tree_)
        return json.dumps(tree_dict)

    def _as_json(self, node: Node) -> dict:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node.left is None and node.right is None:
            # Leaf node
            return {
                "value": int(node.value),
                "n_samples": int(node.n_samples),
                "mse": float(round(node.mse, 2)),
            }
        else:
            # Internal node
            return {
                "feature": int(node.feature),
                "threshold": int(node.threshold),
                "n_samples": int(node.n_samples),
                "mse": float(round(node.mse, 2)),
                "left": self._as_json(node.left),
                "right": self._as_json(node.right),
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        y_pred = np.array([self._predict_one_sample(sample) for sample in X])
        return y_pred

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        node = self.tree_
        while node.left is not None and node.right is not None:
            if features[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

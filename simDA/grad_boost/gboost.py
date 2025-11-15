import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
        subsample_size = 0.5,
        replace = True
    ):
        self.trees_ = []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.base_pred_ = None
        self.subsample_size = subsample_size
        self.replace = replace

    def _mse(self, y_true, y_pred):
        """Mean squared error loss function and gradient."""
        loss = np.mean(np.square(y_pred - y_true))
        grad = y_pred - y_true
        return loss, grad

    def _mae(self, y_true, y_pred):
        """Mean absolute error loss function and gradient."""
        loss = np.mean(np.abs(y_pred - y_true))
        grad = np.sign(y_pred - y_true) / y_true.size
        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_= np.mean(y)
        if isinstance(self.loss, str):
            loss_func = getattr(self, "_" + self.loss)
        elif callable(self.loss):
            loss_func = self.loss
        else:
            raise TypeError("Loss must be string or callable")

        current_pred = np.full_like(y, self.base_pred_, dtype=float)

        for _ in range(self.n_estimators):
            _, grad = loss_func(y, current_pred)

            model = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split)
            X_s, grad_sub = self._subsample(X,grad)
            model.fit(X_s, -grad_sub)
            y_pred = model.predict(X)
            current_pred += self.learning_rate * y_pred
            loss, _ = loss_func(y, y_pred * self.learning_rate)
            if self.verbose:
                print(loss)
            self.trees_.append(model)

        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        y_pred = np.full(X.shape[0], self.base_pred_)

        for tree in self.trees_:
            y_pred += tree.predict(X)*self.learning_rate

        return y_pred

    def _subsample(self, X, y):
        n_samples = X.shape[0]
        n_subsample = int(self.subsample_size * n_samples)

        X_subsampled, y_subsampled = resample(X,y,replace=self.replace,n_samples=n_subsample)

        return X_subsampled, y_subsampled

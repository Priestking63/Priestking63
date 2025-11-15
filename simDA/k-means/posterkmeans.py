from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ImageKMeans:
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to pixel array"""
        height, width, channels = image.shape
        X = image.reshape(height * width, channels)
        return X

    def _init_centroids(self, X: np.ndarray) -> None:
        """Select N random samples as initial centroids"""
        if isinstance(self.init, str):
            if self.init != "random":
                raise ValueError("Init string must be 'random'")
            np.random.seed(self.random_state)
            if len(X) < self.n_clusters:
                raise ValueError("Number of samples less than n_clusters")
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids_ = X[indices]
        elif isinstance(self.init, np.ndarray):
            if self.init.shape != (self.n_clusters, 3):
                raise ValueError("Init array must have shape (n_clusters, 3)")
            if not np.issubdtype(self.init.dtype, np.number):
                raise ValueError("Init must be numeric")
            if np.any(self.init < 0) or np.any(self.init > 255):
                raise ValueError("Centroid values must be between 0 and 255 inclusive")
            unique = np.unique(self.init, axis=0)
            if unique.shape[0] < self.n_clusters:
                raise ValueError("Centroids must be unique")
            self.centroids_ = self.init.copy()
        else:
            raise TypeError("Init must be 'random' or a numpy ndarray")

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        diff = X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]
        squared_distances = np.sum(diff * diff, axis=2)
        return np.argmin(squared_distances, axis=1)

    def fit(self, image: np.ndarray) -> ImageKMeans:
        """Fit k-means to the image"""
        X = self._image_as_array(image)
        self._init_centroids(X)

        prev_centroids = None

        for iteration in range(self.max_iter):
            y = self._assign_centroids(X)
            prev_centroids = self.centroids_.copy()
            self._update_centroids(X, y)

            if np.allclose(prev_centroids, self.centroids_):
                break

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the labels of the image"""
        X = self._image_as_array(image)
        distances = np.sqrt(
            np.sum((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2, axis=2)
        )
        labels = np.argmin(distances, axis=1)
        return labels.reshape(image.shape[:2])

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return the compressed image"""
        labels = self.predict(image)
        compressed = self.centroids_[labels]
        return compressed.reshape(image.shape).astype(image.dtype)

    def _update_centroids(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the centroids by taking the mean of its samples"""
        counts = np.bincount(y, minlength=self.n_clusters)
        mask = counts > 0
        if np.any(mask):
            sums = np.zeros((self.n_clusters, X.shape[1]))
            np.add.at(sums, y, X)
            self.centroids_[mask] = sums[mask] / counts[mask, np.newaxis]

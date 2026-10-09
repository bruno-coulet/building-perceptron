from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PerceptronClassifier(ClassifierMixin, BaseEstimator):
    """Educational binary Perceptron with a scikit-learn compatible API."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        threshold: float = 0.0,
        activation: str = "step",
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.activation = activation
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = check_X_y(X, y, dtype=float)
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError("PerceptronClassifier attend des labels binaires 0 et 1.")
        if self.learning_rate <= 0 or self.max_iter <= 0:
            raise ValueError("learning_rate et max_iter doivent être strictement positifs.")
        if self.activation not in {"step", "sigmoid", "tanh"}:
            raise ValueError("activation doit être 'step', 'sigmoid' ou 'tanh'.")

        generator = np.random.default_rng(self.random_state)
        self.weights_ = np.zeros(X.shape[1], dtype=float)
        self.bias_ = 0.0
        self.errors_ = []
        self.error_rates_ = []
        self.losses_ = []
        self.weights_history_ = []
        self.bias_history_ = []

        for _ in range(self.max_iter):
            indices = generator.permutation(len(X)) if self.shuffle else np.arange(len(X))
            errors = 0
            for index in indices:
                activation = float(X[index] @ self.weights_ + self.bias_)
                output = self._activate(np.asarray([activation - self.threshold]))[0]
                prediction = int(output >= 0.5)
                if self.activation == "step":
                    update = self.learning_rate * (y[index] - prediction)
                else:
                    derivative = output * (1 - output) if self.activation == "sigmoid" else 0.5 * (1 - (2 * output - 1) ** 2)
                    update = self.learning_rate * (y[index] - output) * derivative
                if update:
                    self.weights_ += update * X[index]
                    self.bias_ += update
                    errors += 1
            outputs = self._activate(X @ self.weights_ + self.bias_ - self.threshold)
            epoch_errors = int(np.count_nonzero((outputs >= 0.5).astype(int) != y))
            self.errors_.append(epoch_errors)
            self.error_rates_.append(float(epoch_errors / len(y)))
            if self.activation == "step":
                signed_labels = 2 * y - 1
                margins = signed_labels * (X @ self.weights_ + self.bias_ - self.threshold)
                loss = np.maximum(0.0, -margins).mean()
            else:
                loss = np.mean((y - outputs) ** 2)
            self.losses_.append(float(loss))
            self.weights_history_.append(self.weights_.copy())
            self.bias_history_.append(float(self.bias_))
            if epoch_errors == 0:
                break

        self.n_iter_ = len(self.errors_)
        self.classes_ = classes
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["weights_", "bias_"])
        X = check_array(X, dtype=float)
        return X @ self.weights_ + self.bias_ - self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._activate(self.decision_function(X)) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = self._activate(self.decision_function(X))
        return np.column_stack([1 - positive, positive])

    def _activate(self, scores: np.ndarray) -> np.ndarray:
        if self.activation == "step":
            return (scores >= 0).astype(float)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
        return (np.tanh(scores) + 1.0) / 2.0

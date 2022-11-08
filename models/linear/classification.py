import numpy as np
import pandas as pd
from typing import Union
from scipy.special import expit


class LogisticRegression:
    """
    Logistic regression for classification task. Implements full gradient descent.

    Attributes:
        penalty: 'none', 'l1' or 'l2', default='none'
            Type of regularizazation

        fit_intercept: bool, default=True
            Either to add column of 1's to dataset

        alpha: float > 0, default=1
            Regularization coefficient

        n_iterations: int > 0, default=1
            Number of steps for gradient descent

        random_state: int, default=None
            Random seed

    Methods:
        fit(x: Union[np.ndarray, pd.DataFrame, pd.Series], y: Union[np.ndarray, pd.DataFrame, pd.Series])
            Fitting model to the training data

        predict(x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
            Predict target variable based on test data

        predict_proba(x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
            Predict probabilitis for each class for test data

    """

    def __init__(self, penalty='none', fit_intercept=True, alpha=1, n_iterations=1, random_state=None):
        self._penalty = penalty
        self._fit_intercept = fit_intercept
        self._alpha = alpha
        self._n_iterations = n_iterations
        self._random_state = random_state
        self.coef_ = None

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series],
            y: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fitting model to the training data

        """

        if self._random_state is not None:
            np.random.seed(self._random_state)

        x = np.array(x)
        if self._fit_intercept:
            x = np.append(x, [[1]] * x.shape[0], axis=1)
        y = np.array(y)

        self.coef_ = self.__gradient_descent(x, y)

    def predict(self, x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict target variable based on test data

        """
        x = np.array(x)
        if self._fit_intercept:
            x = np.append(x, [[1]] * x.shape[0], axis=1)
        y = np.sign(x @ self.coef_)
        y = np.where(y > 0, y, y + 1)

        return y

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict probabilitis for each class for test data

        """
        x = np.array(x)
        if self._fit_intercept:
            x = np.append(x, [[1]] * x.shape[0], axis=1)
        y = expit(x @ self.coef_)

        return y

    @staticmethod
    def __calculate_loss(x: np.ndarray, y: np.ndarray, w: np.ndarray, batch: np.ndarray) -> float:
        loss = 0
        epsilon = 10**(-5)
        for sample in batch:
            current = y[sample] * np.log(expit(x[sample] @ w) + epsilon) + (1 - y[sample]) * np.log(1 - expit(x[sample] @ w) + epsilon)
            loss += current
        loss /= -batch.size

        return loss

    def __calculate_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, batch: np.ndarray) -> list:
        gradients = list()
        m = x.shape[1]

        if self._penalty == 'l2':
            def regularization(weight):
                return self._alpha * weight
        elif self._penalty == 'l1':
            def regularization(weight):
                return self._alpha * np.sign(weight)
        elif self._penalty == 'none':
            def regularization(weight):
                return 0

        for i in range(m):
            certain_gradient = 0

            if self._fit_intercept:
                if i == m - 1:
                    for sample in batch:
                        certain_gradient += float(x[sample][i] * (expit(x[sample] @ w) - y[sample]))
                    gradients.append(certain_gradient / batch.size)
                else:
                    for sample in batch:
                        certain_gradient += float(x[sample][i] * (expit(x[sample] @ w) - y[sample]) + regularization(w[i]))
                    gradients.append(certain_gradient / batch.size)
            else:
                for sample in batch:
                    certain_gradient += float(x[sample][i] * (expit(x[sample] @ w) - y[sample]) + regularization(w[i]))
                gradients.append(certain_gradient / batch.size)

        return gradients

    def __gradient_descent(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        w = np.ones(x.shape[1])

        for step in range(1, self._n_iterations):
            gradients = self.__calculate_gradient(x, y, w, np.arange(0, x.shape[0]))
            w -= (1 / step) * np.array(gradients)

        return w

import pandas as pd
import numpy as np


class KNNClassifier:
    """
    K-nearest neigbors metric classification algorithm.

    Attributes:
        k: int>0
            Number of neighbors

    Methods:
        fit(x: pd.DataFrame, y: pd.Series)
            Fit training data (just load it in memory)

        predict(x: pd.DataFrame) -> np.ndarray
            Predict target variable fot test data
    """
    def __init__(self, k: int):
        self._k = k
        self._x = None
        self._y = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        Fit (memorize) training data

        """
        self._x = x
        self._y = y

    def __calculate_distance(self, a: np.ndarray, b: np.ndarray) -> np.float64:
        return np.linalg.norm(a - b)

    def __calculate_kernel(self, r: float) -> float:
        return 0.75 * (1 - r**2)

    def predict(self, x: pd.DataFrame):
        """
        Make a prediction for test data

        """
        prediction = list()

        for i in range(x.shape[0]):
            distances_and_classes = list()

            for j in range(self._x.shape[0]):
                distances_and_classes.append(
                    (self.__calculate_distance(x.iloc[i].values, self._x.iloc[j].values), self._y.iloc[j].values[0]))

            distances_and_classes.sort()
            nearest_elements = distances_and_classes[:self._k + 1]
            nearest_with_kernel = [(self.__calculate_kernel(pair[0] / nearest_elements[self._k][0]), pair[1])
                                   for pair in nearest_elements][:-1]

            one, zero = 0, 0
            for pair in nearest_with_kernel:
                if pair[1] == 0:
                    zero += pair[0]
                else:
                    one += pair[0]

            prediction.append(0 if zero > one else 1)

        return prediction
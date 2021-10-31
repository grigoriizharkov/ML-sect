from typing import Union
import numpy as np
from abc import ABCMeta, abstractmethod
import pandas as pd


class LinearModel(metaclass=ABCMeta):

    def __init__(self, gradient: bool, fit_intercept=True, convergence_rate=0.01, forgetting_rate=0.01, random_state=None):
        self._gradient = gradient
        self._fit_intercept = fit_intercept
        self._convergence_rate = convergence_rate
        self._forgetting_rate = forgetting_rate
        self._random_state = random_state
        self.coef_ = None

    def fit(self, x: Union[np.ndarray, pd.DataFrame, pd.Series],
            y: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        if self._random_state is not None:
            np.random.seed(self._random_state)

        x = np.array(x)
        if self._fit_intercept:
            x = np.append(x, [[1]] * x.shape[0], axis=1)
        y = np.array(y)

        if self._gradient:
            w = self.__gradient_descent(x, y)
        else:
            w = self._solve_accurate(x, y)

        self.coef_ = w

        return w

    def predict(self, x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        x = np.array(x)
        if self._fit_intercept:
            x = np.append(x, [[1]] * x.shape[0], axis=1)
        y = x @ self.coef_

        return y

    @abstractmethod
    def _solve_accurate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _regularization(self, weight: float) -> float:
        pass

    @staticmethod
    def __calculate_loss(x: np.ndarray, y: np.ndarray, w: np.ndarray, batch: np.ndarray) -> float:
        """
        Работает как для стохастического, так и для мини-батча и полного градиентоного спуска.
        Параметр batch - список порядковых номеров объектов из матрицы объекты-признаки, по которым считается ошибка.
        Так, для стохастического градиентного спуска batch будет представлять из себя массив одного элемента - порядкового номера строки (объекта),
        по которому следует считать среднюю по данной подвыборке ошибку.
        Для мини-батча - некоторую подвыборку порядковых номеров строк матрицы объекты-признаки, по которым будет рассчитываться ошибка.
        Для полного градиентного спуска - полный список (от 0 до количества строк матрицы - 1) порядковых номеров строк матрицы - считается средняя ошибка на всей выборке.
        """
        loss = 0
        for sample in batch:
            loss += (x[sample] @ w - y[sample]) ** 2
        loss /= batch.size

        return loss

    def __calculate_gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, batch: np.ndarray) -> list:
        """
        Работает как для стохастического, так и для мини-батча и полного градиентоного спуска.
        Параметр batch - список порядковых номеров объектов из матрицы объекты-признаки, по которым считается градиент.
        Так, для стохастического градиентного спуска batch будет представлять из себя массив одного элемента - порядкового номера строки (объекта),
        по которому следует считать градиент.
        Для мини-батча - некоторую подвыборку порядковых номеров строк матрицы объекты-признаки, по которым будет рассчитываться градиент.
        Для полного градиентного спуска - полный список (от 0 до количества строк матрицы - 1) порядковых номеров строк матрицы - градиент считается на всей выборке.
        """

        gradients = list()
        m = x.shape[1]

        for i in range(m):
            certain_gradient = 0

            if self._fit_intercept:
                if i == m - 1:
                    for sample in batch:
                        certain_gradient += float(x[sample][i] * (x[sample] @ w - y[sample]))
                    gradients.append(certain_gradient * 2 / batch.size)
                else:
                    for sample in batch:
                        certain_gradient += float(
                            x[sample][i] * (x[sample] @ w - y[sample]) + self._regularization(w[i]))
                    gradients.append(certain_gradient * 2 / batch.size)
            else:
                for sample in batch:
                    certain_gradient += float(x[sample][i] * (x[sample] @ w - y[sample]) + self._regularization(w[i]))
                gradients.append(certain_gradient * 2 / batch.size)

        return gradients

    def __gradient_descent(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Полынй градиентный спуск, сходимость которого задается в параметрах модели (рассчитывается через ошибку на текущем и предыдущем шаге).
        В качестве коэффициента при антиградиенте используется простейший динамический шаг = 1 / номер итерации.
        """

        w = np.ones(x.shape[1])  # инициализируем веса
        q = self.__calculate_loss(x, y, w, np.arange(0, x.shape[0]))  # считаем потерю для первоначального приближения

        gradients = self.__calculate_gradient(x, y, w, np.arange(0, x.shape[0]))  # вычисляем вектор градиентов
        w = w - np.array(gradients)  # делаем шаг - обновляем веса

        q_new = self.__calculate_loss(x, y, w, np.arange(0, x.shape[0]))  # считаем полную потерю для новых весов

        step = 2  # задаем переменную для величины шага
        while abs(q_new - q) > self._convergence_rate:
            q = q_new  # храним ошибки для текущих и предыдущих значений вектора весов - нужно для сходимости

            gradients = self.__calculate_gradient(x, y, w, np.arange(0, x.shape[0]))
            w = w - (1 / step) * np.array(gradients)

            q_new = self.__calculate_loss(x, y, w, np.arange(0, x.shape[0]))
            step += 1  # инкремент переменной градиентного шага

        return w


class LinearRegression(LinearModel):

    def _solve_accurate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.solve(x.T @ x, x.T @ y)

    def _regularization(self, weight: float) -> float:
        return 0


class Ridge(LinearModel):

    def __init__(self, alpha, *args, **kwargs):
        self._alpha = alpha
        super().__init__(*args, **kwargs)

    def _solve_accurate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.solve((x.T @ x + self._alpha * np.eye(x.shape[1])), x.T @ y)

    def _regularization(self, weight: float) -> float:
        return self._alpha * weight


class Lasso(LinearModel):

    def __init__(self, alpha, *args, **kwargs):
        self._alpha = alpha
        super().__init__(*args, **kwargs)

    def _solve_accurate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def _regularization(self, weight: float) -> float:
        return self._alpha * np.sign(weight)

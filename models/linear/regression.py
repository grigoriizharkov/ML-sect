import numpy as np
from abc import ABCMeta, abstractmethod


class LinearModel(metaclass=ABCMeta):

    def __init__(self, gradient, fit_intercept=True, convergence_rate=0.01, forgetting_rate=0.01, random_state=None):
        self._gradient = gradient
        self._fit_intercept = fit_intercept
        self._convergence_rate = convergence_rate
        self._forgetting_rate = forgetting_rate
        self._random_state = random_state
        self.coef_ = None

    def fit(self, x, y):
        if self._random_state is not None:
            np.random.seed(self._random_state)

        x = np.array(x)
        if self._fit_intercept:
            x = np.append(x, [[1]] * x.shape[0], axis=1)
        y = np.array(y)

        if self._gradient:
            w = self.gradient_descent(x, y)
        else:
            w = self.solve_accurate(x, y)

        self.coef_ = w

        return w

    def predict(self, x):
        x = np.array(x)
        if self._fit_intercept:
            x = np.append(x, [[1]] * x.shape[0], axis=1)
        y = x @ self.coef_

        return y

    @abstractmethod
    def solve_accurate(self, x, y):
        pass

    @abstractmethod
    def calculate_gradient(self, x, y, w, batch):
        pass

    @staticmethod
    def calculate_loss(x, y, w, batch):
        loss = 0
        for sample in batch:
            loss += (x[sample] @ w - y[sample]) ** 2
        loss /= batch.size

        return loss

    def gradient_descent(self, x, y):
        w = np.ones(x.shape[1])  # инициализируем веса
        q = self.calculate_loss(x, y, w, np.arange(0, x.shape[0]))  # считаем потерю для первоначального приближения
        # sample = np.random.randint(0, X.shape[0])  # случайно выбираем объект, на котором будем считать градиент
        # batch = np.random.choice(X.shape[0], 50, replace=False)

        # gradients = self.calculate_batch_gradient(X, y, w, batch)
        gradients = self.calculate_gradient(x, y, w, np.arange(0, x.shape[0]))  # вычисляем вектор градиентов
        # gradients = self.calculate_stochastic_gradient(X, y, w, sample)  # считаем вектор градиентов только для одного объекта
        w = w - np.array(gradients)  # делаем шаг - обновляем веса
        q_new = self.calculate_loss(x, y, w, np.arange(0, x.shape[0]))  # считаем полную потерю для новых весов

        step = 2  # задаем переменную для величины шага
        while abs(q_new - q) > self._convergence_rate:
            q = q_new  # храним ошибки для текущих и предыдущих значений вектора весов - нужно для сходимости
            # batch = np.random.choice(X.shape[0], 50, replace=False)

            # gradients = self.calculate_batch_gradient(X, y, w, batch)
            gradients = self.calculate_gradient(x, y, w, np.arange(0, x.shape[0]))
            # sample = np.random.randint(0, X.shape[0])
            # epsilon = self.calculate_certain_loss(X, y, w, sample)  # считаем ошибку на конкретном выбранном объекте

            # gradients = self.calculate_stochastic_gradient(X, y, w, sample)
            w = w - (1 / step) * np.array(gradients)

            # q_new = epsilon * self._forgetting_rate + (
            #         1 - self._forgetting_rate) * q  # считаем ошибку для новых весов (скользящее среднее)
            q_new = self.calculate_loss(x, y, w, np.arange(0, x.shape[0]))

            step += 1  # инкремент переменной градиентного шага

        return w


class LinearRegression(LinearModel):

    def solve_accurate(self, x, y):
        w = np.linalg.solve(x.T @ x, x.T @ y)
        return w

    def calculate_gradient(self, x, y, w, batch):
        gradients = list()
        m = x.shape[1]
        for i in range(m):
            certain_gradient = 0
            for sample in batch:
                certain_gradient += float(x[sample][i] * (x[sample] @ w - y[sample]))
            gradients.append(certain_gradient * 2 / batch.size)

        return gradients


class Ridge(LinearModel):

    def __init__(self, alpha, *args, **kwargs):
        self._alpha = alpha
        super().__init__(*args, **kwargs)

    def solve_accurate(self, x, y):
        return np.linalg.solve((x.T @ x + self._alpha * np.eye(x.shape[1])), x.T @ y)

    def calculate_gradient(self, x, y, w, batch):
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
                        certain_gradient += float(x[sample][i] * (x[sample] @ w - y[sample]) + self._alpha * w[i])
                    gradients.append(certain_gradient * 2 / batch.size)
            else:
                for sample in batch:
                    certain_gradient += float(x[sample][i] * (x[sample] @ w - y[sample]) + self._alpha * w[i])
                gradients.append(certain_gradient * 2 / batch.size)

        return gradients


class Lasso(LinearModel):

    def __init__(self, alpha, *args, **kwargs):
        self._alpha = alpha
        super().__init__(*args, **kwargs)

    def solve_accurate(self, x, y):
        raise Exception("Impossible to solve accurate for L1 regularization!")

    def calculate_gradient(self, x, y, w, batch):
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
                        certain_gradient += float(x[sample][i] * (x[sample] @ w - y[sample]) + self._alpha * np.sign(w[i]))
                    gradients.append(certain_gradient * 2 / batch.size)
            else:
                for sample in batch:
                    certain_gradient += float(x[sample][i] * (x[sample] @ w - y[sample]) + self._alpha * np.sign(w[i]))
                gradients.append(certain_gradient * 2 / batch.size)

        return gradients

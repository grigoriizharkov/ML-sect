import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class Node:
    def __init__(self, x: pd.DataFrame, y: pd.Series):
        self.x = x
        self.y = y
        self.right = None
        self.left = None
        self.predicat = None

    def display_tree(self):
        lines, *_ = self.__display()
        for line in lines:
            print(line)

    def __display(self):
        if self.right is None and self.left is None:
            line = f'[{self.predicat[0]} < {self.predicat[1]}, {self.y.shape[0]} sample(s)]' \
                if self.predicat[0] is not None else f'[All resolved, {self.y.shape[0]} sample(s)]'
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        left, n, p, x = self.left.__display()
        right, m, q, y = self.right.__display()
        s = f'[{self.predicat[0]} < {self.predicat[1]}, {self.y.shape[0]} sample(s)]' \
            if self.predicat[0] is not None else f'[All resolved, {self.y.shape[0]} sample(s)] '
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


class DTClassifier:
    def __init__(self, stopping: str, min_sample_leaf=5, max_leaf_number=5):
        if stopping not in ['sample', 'purity', 'number']:
            raise ValueError('Incorrect stopping criteria, choose one of "sample", "purity" or "number"')
        if stopping == 'sample' and min_sample_leaf < 1:
            raise ValueError('Inccorect value of "min_sample_leaf" for "sample" criteria')
        if stopping == 'number' and max_leaf_number < 1:
            raise ValueError('Inccorect value of "max_leaf_number" for "number" criteria')

        self._stopping = stopping
        self._min_sample_leaf = min_sample_leaf
        self._max_leaf_number = max_leaf_number
        self._leaf_number = 1
        self._sample_leaf = 0
        self._root = None

    def __stop(self, leaf_to_check=None):
        if self._stopping == 'sample':
            return self._min_sample_leaf <= leaf_to_check.shape[0]
        elif self._stopping == 'purity':
            return True
        elif self._stopping == 'number':
            return self._leaf_number <= self._max_leaf_number

    def __calculate_entropy(self, y: pd.Series):
        _, counts = np.unique(y, return_counts=True)

        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))

        return entropy

    def __calculate_loss(self, y: pd.Series, y_left: pd.Series, y_right: pd.Series):
        return self.__calculate_entropy(y) - y_left.shape[0] / y.shape[0] * self.__calculate_entropy(y_left)\
            - y_right.shape[0] / y.shape[0] * self.__calculate_entropy(y_right)

    def __choose_predicat(self, x: pd.DataFrame, y: pd.Series):
        q_max = 0
        x_left_best, x_right_best = pd.DataFrame(), pd.DataFrame()
        y_left_best, y_right_best = pd.Series(), pd.Series()
        predicat = [None, None]

        if self.__calculate_entropy(y) == 0:
            return x_left_best, y_left_best, x_right_best, y_right_best, predicat

        for column in x.columns:
            for value in x[column]:
                x_left, x_right = x[x[column] < value], x[x[column] >= value]
                y_left, y_right = y.loc[list(x_left.index.values)], y.loc[list(x_right.index.values)]

                q = self.__calculate_loss(y, y_left, y_right)
                if q >= q_max:
                    q_max = q
                    x_left_best, x_right_best = x_left, x_right
                    y_left_best, y_right_best = y_left, y_right
                    predicat = [column, value]

        return x_left_best, y_left_best, x_right_best, y_right_best, predicat

    def __split(self, node: Node):
        x_left, y_left, x_right, y_right, predicat = self.__choose_predicat(node.x, node.y)
        node.predicat = predicat

        if len(x_left) == 0 and len(x_right) == 0:
            return node
        else:
            self._leaf_number += 1

            if self.__stop(y_right) and self.__stop(y_left):
                node.left = Node(x_left, y_left)
                node.right = Node(x_right, y_right)

                self.__split(node.left)
                self.__split(node.right)
            else:
                return self._root

        return self._root

    def fit(self, x: pd.DataFrame, y: pd.Series):
        node = Node(x, y)
        self.__split(node)
        self._root = node
        return node

    def __choose_leaf(self, sample: pd.Series):
        node = self._root
        description = list()

        while node:
            if node.left is None and node.right is None:
                return np.argmax(np.bincount(node.y[0].to_list())), description

            if sample.loc[node.predicat[0]] < node.predicat[1]:
                description.append(f'{node.predicat[0]} < {node.predicat[1]}')
                node = node.left
            else:
                description.append(f'{node.predicat[0]} >= {node.predicat[1]}')
                node = node.right

    def predict(self, x: pd.DataFrame):
        predictions = list()
        descriptions = list()

        for i in range(x.shape[0]):
            prediction, description = self.__choose_leaf(x.iloc[i])
            predictions.append(prediction)
            descriptions.append(description)

        return predictions, descriptions

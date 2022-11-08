import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RFClassifier:
    """
    Random forest classifier algorithm with entropy splitting criteria

    Attributes:
        n_estimators: int>0
            Number of trees in the forest

        max_depth: int>0, default=None
            Maximum depth of each tree

        min_samples_leaf: int>0, default=1
            Minimum number of samples in the lead

        max_leaf_nodes: int>0, default=None
            Maximum number of leafs in the tree

    Methods:
         fit(x: pd.DataFrame, y: pd.Series) -> Node
            Fitting train data and building the tree

        predict(x: pd.DataFrame) -> list, list
            Predicts class for new test data

        predict_proba(x: pd.DataFrame) -> np.ndarray
            Predicts probability of each class

    """
    def __init__(self, n_estimators: int, max_depth=None, min_samples_leaf=1, max_leaf_nodes=None):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._max_leaf_nodes = max_leaf_nodes
        self._trees = list()

    def fit(self, x: pd.DataFrame, y: pd.Series):
        for i in range(self._n_estimators):
            x_i = x.iloc[np.random.randint(0, x.shape[0], size=x.shape[0])]
            k = int(np.sqrt(x.shape[1]))

            tree_i = DecisionTreeClassifier(criterion='entropy',
                                            max_depth=self._max_depth, min_samples_leaf=self._min_samples_leaf,
                                            max_leaf_nodes=self._max_leaf_nodes, max_features=k)
            tree_i.fit(x_i, y)

            self._trees.append(tree_i)

    def predict(self, x: pd.DataFrame):
        pred = list()

        for i in range(self._n_estimators):
            pred.append(self._trees[i].predict(x))

        prediction = list()
        for i in range(pred[0].shape[0]):
            pred_i = np.mean([pred[j][i] for j in range(len(pred))])
            prediction.append(0 if pred_i < 0.5 else 1)

        return prediction

    def predict_proba(self, x: pd.DataFrame):
        pred = np.ndarray((60, 2500, 2))

        for i in range(self._n_estimators):
            pred[i] = (self._trees[i].predict_proba(x))

        return np.mean(pred, axis=0)






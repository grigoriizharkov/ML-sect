Python: module neighbors.KNNClassifier 


[KNNClassifier](#KNNClassifier)(k: int)  
   
K-nearest neigbors metric classification algorithm.  
   
Attributes:  
    k: int>0  
        Number of neighbors  
   
Methods:  
    [fit](#KNNClassifier-fit)(x: pd.DataFrame, y: pd.Series)  
        Fit training data (just load it in memory)  
   
    [predict](#KNNClassifier-predict)(x: pd.DataFrame) -> np.ndarray  
        Predict target variable fot test data 

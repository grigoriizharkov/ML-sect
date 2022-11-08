Python: class LinearModel 

   
**linear.regression.LinearModel** = class LinearModel([builtins.object](builtins.html#object))

linear.regression.LinearModel(gradient: bool, fit\_intercept=True, convergence\_rate=0.01, forgetting\_rate=0.01, random\_state=None)  
   
Linear regression for classification task. Implements gradient descent or matrix solution  
   
Attributes:  
    gradient: bool  
        Either to use gradient descent or matrix solution  
   
    fit\_intercept: bool, default=True  
        Either to add column of 1's to dataset  
   
    convergence\_rate: float > 0, default=0.01  
        Rate for stopping criteria  
   
    forgetting\_rate: float > 0, default=0.01  
        Rate for forgetting previous weight's vadlue in gradient descent  
   
    random\_state: int, default=None  
        Random seed  
   
Methods:  
    [fit](#linears.regression.LinearModel-fit)(x: Union\[np.ndarray, pd.DataFrame, pd.Series\], y: Union\[np.ndarray, pd.DataFrame, pd.Series\])  
        Fitting model to the training data  
   
    [predict](#linears.regression.LinearModel-predict)(x: Union\[np.ndarray, pd.DataFrame, pd.Series\]) -> np.ndarray:  
        Predict target variable based on test data
        
        
Python: class LogisticRegression 

   
**linear.classification.LogisticRegression** = class LogisticRegression([builtins.object](builtins.html#object))

linear.classification.LogisticRegression(penalty='none', fit\_intercept=True, alpha=1, n\_iterations=1, random\_state=None)  
   
Logistic regression for classification task. Implements full gradient descent.  
   
Attributes:  
    penalty: 'none', 'l1' or 'l2', default='none'  
        Type of regularizazation  
   
    fit\_intercept: bool, default=True  
        Either to add column of 1's to dataset  
   
    alpha: float > 0, default=1  
        Regularization coefficient  
   
    n\_iterations: int > 0, default=1  
        Number of steps for gradient descent  
   
    random\_state: int, default=None  
        Random seed  
   
Methods:  
    [fit](#linears.classification.LogisticRegression-fit)(x: Union\[np.ndarray, pd.DataFrame, pd.Series\], y: Union\[np.ndarray, pd.DataFrame, pd.Series\])  
        Fitting model to the training data  
   
    [predict](#linears.classification.LogisticRegression-predict)(x: Union\[np.ndarray, pd.DataFrame, pd.Series\]) -> np.ndarray:  
        Predict target variable based on test data  
   
    [predict\_proba](#linears.classification.LogisticRegression-predict_proba)(x: Union\[np.ndarray, pd.DataFrame, pd.Series\]) -> np.ndarray:  
        Predict probabilitis for each class for test data
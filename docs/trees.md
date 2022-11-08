Python: module trees.DTClassifier 

class **DTClassifier**([builtins.object](builtins.html#object))



Decision tree classifier algorithm with entropy splitting criteria  
   
Attributes:  
    stopping: 'sample', 'purity' or 'number'  
        Stopping criteria  
   
    min\_sample\_leaf: int>0, default=5  
        Minimum number of samples in the lead  
   
    max\_leaf\_number: int>0, default=5  
        Maximum number of leafs in the tree  
   
Methods:  
     [fit](#DTClassifier-fit)(x: pd.DataFrame, y: pd.Series) -> [Node](#Node)  
        Fitting train data and building the tree  
   
    [predict](#DTClassifier-predict)(x: pd.DataFrame) -> list, list  
        Predict class for new test data 

 


class **Node**([builtins.object](builtins.html#object))



Simple [Node](#Node) class.  
   
Attributes:  
    x: pd.DataFrame  
    y: pd.Series  
   
Methods:  
    [display\_tree](#Node-display_tree)()  
        Visualize tree in console 


Python: module trees.RFClassifier 



class **RFClassifier**([builtins.object](builtins.html#object))



Random forest classifier algorithm with entropy splitting criteria  
   
Attributes:  
    n\_estimators: int>0  
        Number of trees in the forest  
   
    max\_depth: int>0, default=None  
        Maximum depth of each tree  
   
    min\_samples\_leaf: int>0, default=1  
        Minimum number of samples in the lead  
   
    max\_leaf\_nodes: int>0, default=None  
        Maximum number of leafs in the tree  
   
Methods:  
     [fit](#RFClassifier-fit)(x: pd.DataFrame, y: pd.Series) -> Node  
        Fitting train data and building the tree  
   
    [predict](#RFClassifier-predict)(x: pd.DataFrame) -> list, list  
        Predicts class for new test data  
   
    [predict\_proba](#RFClassifier-predict_proba)(x: pd.DataFrame) -> np.ndarray  
        Predicts probability of each class 

 


 


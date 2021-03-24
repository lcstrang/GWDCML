import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from warnings import warn

class DataSet():
    """
    Creates training and testing sets and trains different algorithms
    """
    def __init__(self, data, target, train_size = 0.25, predict_frac = 0.0,
                 dropna = False, nparams = None, max_depth = None,
                 max_samples = 0.3, n_estimators = 10):

        self.target = target
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        if _is_num(data):
            data = data
        elif dropna:
            data = data.dropna()

        if nparams is not None:
            targets = data.loc[:, self.target]
            params = data.drop(columns=self.target)
            params = params.sample(n = nparams, axis = "columns")
            data = params.join(targets)

        X = data.copy()
        label_encoder = preprocessing.LabelEncoder()
        data = X.apply(label_encoder.fit_transform)
        self.enc = preprocessing.OneHotEncoder()

        self.training_data, self.validation_data = train_test_split(data,
                                                                    train_size = train_size)
        self.max_samples = int(self.training_data.shape[0]*max_samples)

        self._decision_tree = None
        self._forest = None
    


    @property
    def decision_tree(self):
        if self._decision_tree is not None:
            pass
        else:
            to_train = tree.DecisionTreeClassifier(max_depth = self.max_depth)

            data = self.training_data.copy()
            targets = data.loc[:, self.target]
            samples = data.drop(columns = self.target)
            self._decision_tree = to_train.fit(samples, targets)
        return self._decision_tree

    @property
    def forest(self):
        if self._forest is not None:
            pass
        else:
            to_train = RandomForestClassifier(n_estimators = self.n_estimators,
                                              max_depth = self.max_depth,
                                              max_samples = self.max_samples)
            data = self.training_data.copy()
            targets = data.loc[:, self.target]
            samples = data.drop(columns = self.target)
            self._forest = to_train.fit(samples, targets)
        return self._forest

    def validate_tree(self):
        data = self.validation_data.copy()
        targets = data.loc[:,self.target].copy()
        samples = data.drop(columns = self.target)
        score = self.decision_tree.score(samples, targets)
        # print(f"Automated: {score}")
        # predictions = self.decision_tree.predict(samples)
        # match = [target == predictions[i] for i, target in enumerate(np.array(targets))]
        # print(f"Manual: {score}")
        return score

    def validate_forest(self):
        data = self.validation_data.copy()
        targets = data.loc[:,self.target].copy()
        samples = data.drop(columns = self.target)
        score = self.forest.score(samples, targets)
        print(score)
        return(score)
        

def _is_num(x):
    try:
        x/3
    except TypeError:
        return False
    else:
        return True

def check_training_size(data, nsample, method = "tree", dropna = False, max_depth = None):
    train_size = np.linspace(start = 0, stop = 1, num = nsample+1,
                               endpoint = False)[1:]
    frames = [DataSet(data, target = "RainTomorrow", train_size = t,
                    dropna = dropna, max_depth = max_depth) for t in train_size]
    if method == "tree":
        accuracy = [frame.validate_tree() for frame in frames]
    elif method == "forest": # NB - results very boring for rain data because the single tree is so effective
        accuracy = [frame.validate_forest() for frame in frames]
    return [train_size, accuracy]

def check_n_params(data, dropna = False):
    size = 10
    nparams = [2, 5, 10, 15, 18, 22]
    ensemble = {n : [DataSet(data, target = "RainTomorrow", nparams = n,
                             dropna = dropna) for i in range(size)] for n in nparams}
    accuracies = {key : [frame.validate_tree() for frame in val] for key, val in ensemble.items()}
    accuracy = np.transpose([[key, np.mean(vals)] for key, vals in accuracies.items()])
    return accuracy

def check_max_depth(data, dropna = False):
    depths = [1, 2, 3, 5, 7, 10]
    frames = [DataSet(data, target = "RainTomorrow", dropna = dropna,
                      max_depth = d) for d in depths]
    accuracy = [frame.validate_tree() for frame in frames]
    print(depths)
    print(accuracy)
    return [depths, accuracy]

def check_tree(data):
    #increasing depths beyond one split unnecessary? seems suspicious
    na_inc = check_max_depth(df, dropna = False)
    na_exc = check_max_depth(df, dropna = True)
    # # more params dips, then improves
    na_inc = check_n_params(df, dropna = False)
    na_exc = check_n_params(df, dropna = True)
    #including data is more important as the size shrinks due to dropna
    na_inc = check_training_size(df, 5, dropna = False)
    na_exc = check_training_size(df, 5, dropna = True)

    #TODO plots?
    
if __name__ == "__main__":
    df = pd.read_csv("weather_nopred.csv")
    check_tree(data)


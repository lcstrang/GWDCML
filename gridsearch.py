import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam

df = pd.read_csv("weatherAUS.csv")
df1 = df.copy(deep= True)
# removing all entries with no values in rain tomorrow
df1 = df1[~df1["RainTomorrow"].isna()]


# Replacing Nans with mode values and medians in case of integers.
for i in list(df1):
    if i != "Date":
        if df1[df1[i].isna()].shape[0] !=0:
            if df1[i].dtype == 'O':
                df1[i] = df1[i].replace(np.NaN, df1[i].mode()[0])
            else:
                df1[i] = df1[i].replace(np.NaN, df1[i].mean())

df1["Month"] = pd.to_numeric(df1["Date"].str.split("-", expand = True)[1])

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for i in list(df):
    if df[i].dtype == 'O':
        le.fit(df1[i]) 
        df1[i]=le.transform(df1[i]) 


X = df1.drop('RainTomorrow', axis = 1)
y = df1['RainTomorrow']


scaler = StandardScaler()
scaler.fit(X)
X_transform = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.2, random_state=102)

from keras.wrappers.scikit_learn import KerasClassifier
def model_simple():
    model = Sequential()
    model.add(Dense(1024, input_dim=23, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    opt = Adam(lr=0.001, decay = 0.00001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'mse'])
    model.summary()
    return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=model_simple, verbose=1)
# define the grid search parameters
batch_size = [20, 40, 60, 80, 100, 120, 140]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, keras.utils.to_categorical(y_train, num_classes=2))
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


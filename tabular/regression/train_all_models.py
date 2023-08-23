import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import norm, spearmanr
from sklearn.datasets import load_breast_cancer, load_iris

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler
#from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

#import lime
#import lime.lime_tabular
import shap
import os
from sklearn.base import TransformerMixin
import re
#import logging
#from explanation_wrapper import lime_exp, shap_exp, lpi_exp, ground_truth

from joblib import dump
from tabulate import tabulate
import lime
import shap
import pickle
BASE_PATH ='/home/amirh/code/can_explanations_explain_lam/tabular/regression/data'

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X
            

data_dict = {}

'''data = pd.read_csv('{}/insurance/insurance.csv'.format(BASE_PATH))
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
target = data['charges']
data = data.drop(columns=['region', 'charges'])
data_dict['insurance'] = [data, target, data.columns]'''

df = pd.read_csv('{}/automobile/Automobile_data.csv'.format(BASE_PATH))
df.replace("?", np.nan, inplace = True)

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore = df["bore"].astype("float").mean(axis=0)
df["bore"].replace(np.nan, avg_bore, inplace=True)
avg_stroke = df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
df.dropna(subset=["price"], axis=0, inplace=True)
df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace=True)
df["num-of-doors"].replace(np.nan, "four", inplace=True)
df["num-of-doors"].replace("two", 2, inplace=True)
df["num-of-doors"].replace("four", 4, inplace=True)

# Reset index, because we droped four rows
df.reset_index(drop=True, inplace=True)

# Converting mpg to L/100km as earlier stated
df["num-of-cylinders"].replace(np.nan, "four", inplace=True)
df["num-of-cylinders"].replace("four", 4, inplace=True)
df["num-of-cylinders"].replace("six", 6, inplace=True)
df["num-of-cylinders"].replace("five", 5, inplace=True)
df["num-of-cylinders"].replace("two", 2, inplace=True)
df["num-of-cylinders"].replace("eight", 8, inplace=True)
df["num-of-cylinders"].replace("three", 3, inplace=True)
df["num-of-cylinders"].replace("twelve", 12, inplace=True)
df['city-L/100km'] = 235/df["city-mpg"]

df["highway-L/100km"] = 235/df["highway-mpg"]

# Renaming attribute name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# Converting the data types to proper data format
df[["normalized-losses"]] = df[["normalized-losses"]].astype("float")
df[["stroke", "bore"]] = df[["stroke","bore"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("float")
df[["horsepower","peak-rpm","price"]] = df[["horsepower","peak-rpm","price"]].astype("float")
df[["engine-size","city-mpg","highway-mpg","curb-weight"]] = df[["engine-size","city-mpg","highway-mpg","curb-weight"]].astype("float")
df[["num-of-doors"]] = df[["num-of-doors"]].astype("float")
df[["num-of-cylinders"]] = df[["num-of-cylinders"]].astype("float")

y=df["price"]
x=df.drop(["make", "aspiration","fuel-type","aspiration","body-style","drive-wheels","engine-location","engine-type","fuel-system","city-mpg","highway-mpg","price"],axis=1)

data_dict['automobile'] = [x, y, df.columns]

'''from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
data_dict['housing'] = [housing['data'], housing['target'], housing['feature_names']]'''


additional_ds = ['wineWhite', 'wineRed', 'puma8fh', 'anacalt',  'kin8nm', 'istanbul', 'bank8fm', 'mg', 'treasury', 'kin8fm',
               'quakes', 'kin8fh', 'wizmir', 'bank8fh',  'kin8nh', 'puma8fm', 'bank8nm', 'bank8nh', 'deltaA', 'deltaE', 'mortage']


for ad_ds in additional_ds: 
    d = pd.read_csv('{}/additional_regression/{}.txt'.format(BASE_PATH, ad_ds))
    y = d['REGRESSION']
    x = d.iloc[:, 1:]
    feature_names = x.columns.values
    data_dict[ad_ds] = [x, y, feature_names]


d_info = {}
for data_key in data_dict:
    d_info[data_key] = {'features': data_dict[data_key][2]}
pickle.dump(d_info, open("{}/data_info_v1.p".format(BASE_PATH), "wb"))
sys.exit()
for data_key in data_dict.keys():
    X_train, X_test, y_train, y_test = train_test_split(data_dict[data_key][0], data_dict[data_key][1], random_state=0)

    preproc = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}

    for proc in preproc:
        if not os.path.exists('{}/{}'.format(BASE_PATH, data_key)):
            os.makedirs('{}/{}/'.format(BASE_PATH, data_key))
            
        if not os.path.exists('{}/{}/{}'.format(BASE_PATH, data_key, proc)):
            os.makedirs('{}/{}/{}'.format(BASE_PATH, data_key, proc))

        np.save('{}/{}/{}/x_test.npy'.format(BASE_PATH, data_key, proc), X_test)
        np.save('{}/{}/{}/x_train.npy'.format(BASE_PATH, data_key, proc), X_train)
        np.save('{}/{}/{}/y_test.npy'.format(BASE_PATH, data_key, proc), y_test)
        
        
        lr = Pipeline([
            ('preprocess', preproc[proc]),
            ('to_dense', DenseTransformer()), 
            ('lr', LinearRegression())
        ])

        lr.fit(X_train, y_train)

        dump(lr, '{}/{}/{}/lr_v2.joblib'.format(BASE_PATH, data_key, proc))









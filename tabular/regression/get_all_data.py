
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

#import lime
#import lime.lime_tabular
import shap
import os
from sklearn.base import TransformerMixin

#import logging

from joblib import dump
from tabulate import tabulate

import pickle


def get_unique_cat(data, cat_columns):
    dic = {}
    for s in cat_columns:
        idx = np.argwhere(s == data.columns).flatten()[0]
        dic[idx] = data.loc[:, s].unique()
    return dic

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X

def get_cat_col_idx(data, categorical_columns):
    categorical_columns_idx = []
    for cat in categorical_columns:
        categorical_columns_idx.append(data.columns.get_loc(cat))
    return categorical_columns_idx


def get_all_data():
    base_path ='/home/amirh/code/local_explanations_whitebox/tabular/data'
    data = {}
    anchor_dict = {}
    # 1. Breast Cancer
    breast_cancer = load_breast_cancer()
    features = np.array(breast_cancer.feature_names)
    data['breast_cancer'] = [breast_cancer.data, breast_cancer.target, features, [], []]

 
    # 20. Banking
    banking = pd.read_csv('{}/banking/new_train.csv'.format(base_path))
    y = banking['y'].values
    y[y == 'yes'] = 1
    y[y == 'no'] = 0
    y = y.astype(int)
    banking = banking.drop(['y'], axis=1)
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
           'month', 'day_of_week', 'poutcome']

    features = np.array(banking.columns)
    data['banking'] = [banking, y, features, categorical_columns]

    return data

def generate_data_info(): 
    data = get_all_data()
    d_info = {}
    
    for data_key in data:
        cat_col_idx = get_cat_col_idx(data[data_key][0], data[data_key][3])
        d_info[data_key] = {'features': data[data_key][2], 'cat_feature_name': data[data_key][3], 'cat_feature_idx': cat_col_idx}
        
    pickle.dump(d_info, open("/tf/tabular/data/data_info.p", "wb"))
    
    
if __name__ == "__main__":
    generate_data_info()

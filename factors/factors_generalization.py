from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import lime
import lime.lime_tabular
import shap
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm, spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity, kernel_metrics
from sklearn.metrics import log_loss
import matplotlib
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import sys
sys.path.append('../')
import scipy
from sklearn.preprocessing import KBinsDiscretizer
#import scikit_posthocs
from techniques import get_local_permutation_importance, get_lime_explanations, get_shap_explanations, nbayes_local_explaination, lreg_local_explaination
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import pickle
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import pandas as pd


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X

        
sys.path.append('../tabular/')
from get_all_explanations import lime_exp, shap_exp, lpi_exp, ground_truth

d_info = {}
random_state = 0
num_feat_bin = [4]

i = 0

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X

redundant = int(np.round(0.2* (40 - num_feat_bin[i])))
X, y = make_classification(
    n_features=40, n_redundant=redundant, n_clusters_per_class=2, random_state=0
)


d_info[i] = {'features': [], 'cat_feature_name': [], 'cat_feature_idx': []}

est = KBinsDiscretizer(n_bins=4, encode='onehot', strategy='uniform')
one_hot_data = est.fit_transform(X[:, -num_feat_bin[i]:]).toarray()
not_one_hot = np.arange(0, 40 - num_feat_bin[i])
X = np.append(X[:, not_one_hot], one_hot_data, axis=1)

cat_feature_idx = np.setxor1d(np.arange(X.shape[1]), not_one_hot)

d_info[i]['cat_feature_idx'] = cat_feature_idx
d_info[i]['cat_feature_name'] = cat_feature_idx

for cat in cat_feature_idx: 
    enc = LabelEncoder()
    X[:,cat] = enc.fit_transform(X[:, cat])

features = []
for k in range(X.shape[1]):
    features.append('Feature {}'.format(k))
d_info[i]['features'] = features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def get_random_params(model_name):
    hyperparams = {
        'lreg': {
            'penalty': ['l1', 'l2'], 
            'C': [0.01, 0.1, 0.5, 1, 2],
            'max_iter': [1000, 2000, 10000],
            'solver': ['saga']
        },
        'gbayes': {
            'priors': [[0.3, 0.7], [0.5, 0.5], None],
            'var_smoothing': [0, 0.1, 0.2, 0.01]
        }   
    } 
    
    params = {}
    for h in hyperparams[model_name]:
        vals = hyperparams[model_name][h]
        sel = np.random.choice(vals, 1)[0]
        params[h] = sel
    return params

exp_function = {
        'lime': lime_exp,
        'shap': shap_exp,
        'lpi': lpi_exp,
    }



accuracy_scores_ = {'gbc': [], 'rfc': []}
gts = {}
exps = {}

for i in range(10):
    
    
    
    gts[i] = {'lreg': [], 'gbayes': []}
    exps[i] = {'lreg': {'lpi': [], 'lime': [], 'shap': []}, 'gbayes': {'lpi': [], 'lime': [], 'shap': []}}
    
    params = get_random_params('lreg')
    categorical_encoder_lreg = OneHotEncoder(handle_unknown="ignore")
    preprocessing_lreg = ColumnTransformer([
        ("cat", categorical_encoder_lreg, cat_feature_idx)], remainder='passthrough')
    
    lreg = Pipeline([
        ("preprocess", preprocessing_lreg),
        ('to_dense', DenseTransformer()), 
        ("lreg", LogisticRegression(**params))
    ])
    
    lreg.fit(X_train, y_train)
    accuracy_scores_['lreg'].append(accuracy_score(y_test, lreg.predict(X_test)))
    
    params = get_random_params('gbayes')
    categorical_encoder_gb = OneHotEncoder(handle_unknown="ignore")    
    preprocessing_gb = ColumnTransformer([
        ("cat", categorical_encoder_gb, cat_feature_idx)], remainder='passthrough')
    
    gbayes = Pipeline([
        ("preprocess", preprocessing_gb),
        ('to_dense', DenseTransformer()), 
        ("gbayes", GaussianNB(**params))
    ])
    
    
    gbayes.fit(X_train, y_train)
    accuracy_scores_['gbayes'].append(accuracy_score(y_test, gbayes.predict(X_test)))
    
    features = np.array(features)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,  feature_names = features, verbose=False, mode='classification', discretize_continuous=False)
    
    for j in range(len(X_test)):
    #for j in range(1):
        gts[i]['gbayes'].append(ground_truth(X_test[j], X_train, d_info, gbayes, 0, 'gbayes'))
        gts[i]['lreg'].append(ground_truth(X_test[j], X_train, d_info, lreg, 0, 'lreg'))
        
    for e_names in exp_function.keys():
        for j in range(len(X_test)):
        #for j in range(1):
            exps[i]['lreg'][e_names].append(exp_function[e_names](X_test[j], X_train, d_info, lreg, 0, 'lreg'))
            exps[i]['gbayes'][e_names].append(exp_function[e_names](X_test[j], X_train, d_info, gbayes, 0, 'gbayes'))
    
    print('------------------------------------------------')
    
pickle.dump( exps, open( "./exps_all_generalization.p", "wb" ) )
pickle.dump( gts, open( "./gt_all_generalization.p", "wb" ) )
pickle.dump( accuracy_scores_, open( "./generalization_accuracy_scores.p", "wb" ) )
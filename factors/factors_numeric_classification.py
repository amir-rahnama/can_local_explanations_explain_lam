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
from scipy.stats import norm, spearmanr
from sklearn.metrics.pairwise import cosine_similarity, kernel_metrics
from sklearn.metrics import log_loss
import matplotlib
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import sys
sys.path.append('../')
import scipy
from sklearn.preprocessing import KBinsDiscretizer
import scikit_posthocs
from techniques import get_local_permutation_importance, get_lime_explanations, get_shap_explanations, nbayes_local_explaination, lreg_local_explaination
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import pickle
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
sys.path.append('../tabular/')
from get_all_explanations import lime_exp, shap_exp, lpi_exp, ground_truth

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X
        
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

all_datasets = []

exps = {}
gts = {}
d_info = {}

explained_class = 1
accuracy_scores_ = {'lreg': [], 'gbayes': []}
num_features = np.arange(5, 50, 5)

for i in range(len(num_features)):
    print(i)
    gts[i] = {'lreg': [], 'gbayes': []}
    exps[i] = {'lreg': {'lpi': [], 'lime': [], 'shap': []}, 'gbayes': {'lpi': [], 'lime': [], 'shap': []}}
    
    redundant = int(np.round(0.2* num_features[i]))
    X, y = make_classification(
        n_features=num_features[i], n_redundant=redundant, n_clusters_per_class=2, random_state=0
    )
    
    d_info[i] = {'features': [], 'cat_feature_name': [], 'cat_feature_idx': []}
    
    
    gbayes = Pipeline([
            ('to_dense', DenseTransformer()), 
            ("gbayes", GaussianNB())
        ])

    lreg = Pipeline([
        ('to_dense', DenseTransformer()), 
        ("lreg", LogisticRegression(random_state=0, max_iter=10000))
    ])
    
    features = []
    for k in range(X.shape[1]):
        features.append('Feature {}'.format(k))
    d_info[i]['features'] = features
                 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    
    lreg.fit(X_train, y_train)
    gbayes.fit(X_train, y_train)
    
    print('After', X.shape)
    accuracy_scores_['lreg'].append(accuracy_score(y_test, lreg.predict(X_test)))
    accuracy_scores_['gbayes'].append(accuracy_score(y_test, gbayes.predict(X_test)))
    
    print('---------')
    
    
    '''exp_function = {
        'lime': lime_exp,
        'shap': shap_exp,
        'lpi': lpi_exp,
    }

    features = np.array(features)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,  feature_names = features, verbose=False, mode='classification', discretize_continuous=False)
    
    for j in range(len(X_test)):
        gts[i]['gbayes'].append(ground_truth(X_test[j], X_train, d_info, gbayes, i, 'gbayes'))
        gts[i]['lreg'].append(ground_truth(X_test[j], X_train, d_info, lreg, i, 'lreg'))
        
    
    for e_names in exp_function.keys():
        for j in range(len(X_test)):
            exps[i]['lreg'][e_names].append(exp_function[e_names](X_test[j], X_train, d_info, lreg, i, 'lreg'))
            exps[i]['gbayes'][e_names].append(exp_function[e_names](X_test[j], X_train, d_info, gbayes, i, 'gbayes'))'''


#pickle.dump( exps, open( "./exps_all_numeric.p", "wb" ) )
#pickle.dump( gts, open( "./gt_all_numeric.p", "wb" ) )
pickle.dump( accuracy_scores_, open( "./accuracy_scores_all_numeric.p", "wb" ) )
    
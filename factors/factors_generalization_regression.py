from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons, make_circles, make_classification, make_regression
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
from sklearn.metrics import log_loss, mean_squared_error
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
sys.path.append('../tabular/regression')
from get_explanations import lime_exp, shap_exp, lpi_exp
from sklearn.linear_model import Ridge, LinearRegression

sys.path.append('../tabular')
from get_all_explanations import ground_truth

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
num_feat_bin = [4]

informative = int(np.round(0.8* (40 - num_feat_bin[0])))
X, y = make_regression(
    n_features=40, n_informative=informative, noise=0.1, random_state=0
)

d_info[0] = {'features': [], 'cat_feature_name': [], 'cat_feature_idx': []}

est = KBinsDiscretizer(n_bins=4, encode='onehot', strategy='uniform')
one_hot_data = est.fit_transform(X[:, -num_feat_bin[0]:]).toarray()
not_one_hot = np.arange(0, 40 - num_feat_bin[0])
X = np.append(X[:, not_one_hot], one_hot_data, axis=1)

cat_feature_idx = np.setxor1d(np.arange(X.shape[1]), not_one_hot)
num_feat_idx = np.setxor1d(np.arange(X.shape[1]), cat_feature_idx)

scaler = StandardScaler()

#for num in num_feat_idx:
#    X[:, num] = scaler.fit_transform(X[:, num])
X[:, num_feat_idx] = scaler.fit_transform(X[:, num_feat_idx])
print(cat_feature_idx)

d_info[0]['cat_feature_idx'] = cat_feature_idx
d_info[0]['cat_feature_name'] = cat_feature_idx

# name encode the cat features
for cat in cat_feature_idx: 
    enc = LabelEncoder()
    X[:,cat] = enc.fit_transform(X[:, cat])

accuracy_scores_ = {}


def get_random_params(model_name):
    hyperparams = {
        'lr': {
            'alpha': np.arange(0, 20), 
            'fit_intercept': [True, False],
            'max_iter': [500, 1000, 2000, 10000],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }   
    } 
    
    params = {}
    for h in hyperparams[model_name]:
        vals = hyperparams[model_name][h]
        sel = np.random.choice(vals, 1)[0]
        params[h] = sel
    return params


for i in range(10):
    print(i)
    gts[i] = {'lr': []}
    exps[i] = {'lr': {'lpi': [], 'lime': [], 'shap': []}}
    
    params = get_random_params('lr')
    
    categorical_encoder_lreg = OneHotEncoder(handle_unknown="ignore")
    preprocessing_lreg = ColumnTransformer([
        ("cat", categorical_encoder_lreg, cat_feature_idx)], remainder='passthrough')

    lr = Pipeline([
        ('preprocess', preprocessing_lreg),
        ('to_dense', DenseTransformer()), 
        ('lreg', Ridge(**params))
     ])

            
    features = []
    for k in range(X.shape[1]):
        features.append('Feature {}'.format(k))
    d_info[0]['features'] = features
                 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    
    lr.fit(X_train, y_train)
    
    print('After', mean_squared_error(y_test, lr.predict(X_test)))
    accuracy_scores_['lr'].append(mean_squared_error(y_test, lr.predict(X_test)))
    
    
    print('---------')
    
    exp_function = {
        'lime': lime_exp,
        'shap': shap_exp,
        'lpi': lpi_exp,
    }

    features = np.array(features)
    
    for j in range(len(X_test)):
    #for j in range(1):
        #gts[i]['lr'].append(ground_truth( X_train, X_test[j], lr, d_info[i]))
        gts[i]['lr'].append(ground_truth(X_test[j], X_train, d_info, lr, 0, 'lreg'))
            
    for e_names in exp_function.keys():
        for j in range(len(X_test)):
        #for j in range(1):
            exps[i]['lr'][e_names].append(exp_function[e_names](X_train,X_test[j], lr, d_info[0]))
        
pickle.dump( exps, open( "./exps_generalization_regresion_all.p", "wb" ) )
pickle.dump( gts, open( "./gt_generalization_regresion_all.p", "wb" ) )
pickle.dump( accuracy_scores_, open( "./generalization_regression_accuracy_scores.p", "wb" ) )
    
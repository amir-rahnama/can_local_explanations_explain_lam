import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import pandas
from scipy.stats import norm, spearmanr
import lime
import pandas as pd
import lime.lime_tabular
from sklearn.datasets import make_moons, make_circles, make_classification
import scipy
import scikit_posthocs
import shap
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, kernel_metrics
from tabulate import tabulate
import sys
import pickle
import joblib
from functools import partial
import logging 
import sys
import os
from numpy import linalg as LA
from get_all_explanations import lime_exp, shap_exp, lpi_exp, ground_truth

from sklearn.base import TransformerMixin



class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X
        
BASE_PATH ='/home/amirh/code/local_explanations_whitebox/tabular/data'

logger = logging.getLogger('{}/calculate_robust'.format(BASE_PATH))
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('{}/calculate_robust.log'.format(BASE_PATH))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

#BASE_PATH = '/tf/tabular/'
#preprocessing = ['robust', 'standard', 'minmax']
preprocessing = ['standard']

models = ['lreg', 'gbayes']
exp_types = ['lime', 'shap', 'lpi']
datasets = ['breast_cancer', 'pima_indians', 'banknote', 'iris', 'haberman', 'spambase', 'heart_disease', 'churn', 'hattrick', 'hr', 'audit', 'loan', 'seismic', 'thera', 'titanic', 'donor', 'adult', 'insurance', 'banking', 'attrition' ]
max_threshold = 100

epsilon = 1e-3
folder_name = 'exp_v10'
robust_types = ['insertion', 'deletion']

data_info = pickle.load(open("{}/data_info_v2.p".format(BASE_PATH), "rb"))
explained_class = 1

cutoffs = np.linspace(0.05, 0.5, 10)

robustness = {}
for preproc in preprocessing:
    #robustness[preproc] = {}
    
    for model in models:
        robustness[model] = {}
        
        for exp in exp_types:
            robustness[model][exp] = {}
            for data_key in datasets:
                robustness[model][exp][data_key] = {}
                model_object = joblib.load('{}/{}/{}/{}_v1.joblib'.format(BASE_PATH, data_key, preproc, model))
                has_categories = ('preprocess' == model_object.steps[0][0])
        
                exp_result = np.load('{}/{}/{}/{}/{}_exp_{}_{}.npy'.format(BASE_PATH, data_key, preproc, folder_name,
                                                                  exp, model, preproc), allow_pickle=True)
                x_test = np.load('{}/{}/{}/x_test.npy'.format(BASE_PATH, data_key, preproc),  allow_pickle=True)
                x_train = np.load('{}/{}/{}/x_train.npy'.format(BASE_PATH, data_key, preproc),  allow_pickle=True)
                                
                total_run = x_test.shape[0] if x_test.shape[0] < max_threshold  else max_threshold

                logger.info('{}, {}, {}, {}'.format(data_key, preproc, model, exp))
            
                
                all_cat_features = data_info[data_key]['cat_feature_idx']
                
                for robust_type in robust_types:
                    robustness[model][exp][data_key][robust_type] = {}
                    temp_result = []
                    for i in range(total_run):
                        all_cutoffs = []
                        for j in range(len(cutoffs)):
                            exp_example = exp_result[i]
                            explained_instance = x_test[i]
                            
                            explained_instance_copy = explained_instance.copy()
                            base_pred = model_object.predict_proba(explained_instance.reshape(1, -1))[:, explained_class][0]
                            threshold = int(np.round(cutoffs[j] * exp_example.shape[0]))
    
                            feat_selected = np.abs(exp_example).argsort()[::-1][:threshold]
                            if robust_type == 'insertion':
                                feat_selected = np.setxor1d(np.arange(exp_example.shape[0]), feat_selected)
                            
                            for f in feat_selected:
                                if f in all_cat_features:

                                    all_feat_vals =  x_test[:, f].astype(int)
                                    if not isinstance(all_feat_vals, np.ndarray):
                                        all_feat_vals = all_feat_vals.toarray().flatten()
                                    get_most_common = np.bincount(all_feat_vals).argmax()
                                    explained_instance_copy[f] = get_most_common
                                else:
                                    explained_instance_copy[f] = x_test[:, f].mean()
                            
                            new_pred = model_object.predict_proba(explained_instance_copy.reshape(1, -1))[:, explained_class][0]
                            
                            all_cutoffs.append(np.abs(new_pred - base_pred))
                        temp_result.append(all_cutoffs)
                        
                    robustness[model][exp][data_key][robust_type] =  temp_result                                            
            pickle.dump(robustness, open( "{}/robustness_measure_v11.p".format(BASE_PATH), "wb" ) )
                    
pickle.dump(robustness, open( "{}/robustness_measure_v11.p".format(BASE_PATH), "wb" ) )
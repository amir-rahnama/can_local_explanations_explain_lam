import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import pandas
from scipy.stats import norm, spearmanr, pearsonr
import lime
import pandas as pd
import lime.lime_tabular
from sklearn.datasets import make_moons, make_circles, make_classification
import scipy
import scikit_posthocs
import shap
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, kernel_metrics, cosine_distances
from tabulate import tabulate
import sys
import pickle
from numpy import linalg
import joblib
from sklearn.base import TransformerMixin



class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X

#datasets = ['breast_cancer', 'pima_indians', 'banknote', 'iris', 'haberman', 'spambase', 'heart_disease', 'adult']
datasets = ['hattrick', 'breast_cancer', 'pima_indians', 'banknote', 'iris', 
   'haberman', 'spambase', 'titanic', 'heart_disease', 'churn', 
                 'hr', 'audit', 'loan', 'attrition', 
                'donor', 'seismic', 'thera', 'adult', 'insurance', 'banking']

#datasets = ['loan', 'breast_cancer', 'donor']
#'hattrick',
preprocessing = ['robust', 'standard', 'minmax']
models = ['lreg', 'gbayes']
styles = {'odds': ['lime', 'shap', 'permute'], 'proba': ['lime', 'shap', 'permute']}
#folder_name = 'exp_v9'
#folder_name = 'exp_v13'
folder_name = 'exp_v10'
BASE_PATH ='/home/amirh/code/local_explanations_whitebox/tabular/data'
data_info = pickle.load(open("{}/data_info_v2.p".format(BASE_PATH), "rb"))

measure = {}
for preproc in preprocessing:
    measure[preproc] = {}
    #exp_models = ['lime', 'shap', 'lpi']
    exp_models = ['lime', 'shap', 'lpi']

    for model in models:
        measure[preproc][model] = {}
        for exp_model in exp_models:
            measure[preproc][model][exp_model] = []
            
            for dataset in datasets:
                model_obj = joblib.load('{}/{}/{}/{}_v1.joblib'.format(BASE_PATH, dataset, preproc, model))
                has_categories = ('preprocess' == model_obj.steps[0][0])

                x_test = np.load('{}/{}/{}/x_test.npy'.format(BASE_PATH, dataset, preproc),  allow_pickle=True)
                #local_exp = np.load('{}/{}/{}/{}/gt_exp_{}_{}.npy'.format(BASE_PATH, dataset, preproc, folder_name,
                #                                                  model, preproc), allow_pickle=True)
                local_exp = np.load('{}/{}/{}/{}/gt_exp_{}_{}.npy'.format(BASE_PATH, dataset, preproc, 'exp_v10',
                                                                  model, preproc), allow_pickle=True)
                exp_result = np.load('{}/{}/{}/{}/{}_exp_{}_{}.npy'.format(BASE_PATH, dataset, preproc, folder_name,
                                                                  exp_model, model, preproc), allow_pickle=True)
                
                measure_vals = {'norm': [], 'spearman': [], 'spearman_abs': [], 'cosine': []}
        
                for i in range(exp_result.shape[0]):
                    if exp_model == 'lpi':
                        exp_result_ = exp_result[i]
                    else:
                        exp_result_ = np.multiply(exp_result[i], x_test[i])
                    
                    val_norm = linalg.norm(exp_result_ - local_exp[i])
                    sim_norm = 1 / (val_norm + 1)
                    measure_vals['norm'].append(sim_norm)
                    
                    sr = spearmanr(exp_result_, local_exp[i], axis=1).correlation
                    measure_vals['spearman'].append(sr)
                    
                    sr_abs = spearmanr(np.abs(exp_result_), np.abs(local_exp[i]), axis=1).correlation
                    measure_vals['spearman_abs'].append(sr_abs)
                    
                    cd = cosine_similarity(exp_result_.reshape(1, -1).astype(np.float64), 
                                          local_exp[i].reshape(1, -1).astype(np.float64))[0][0]
                    measure_vals['cosine'].append(cd)
                
                
                measure[preproc][model][exp_model].append(measure_vals)

pickle.dump( measure, open( "{}/exp_accuracy_v10_new.p".format(BASE_PATH), "wb" ) )
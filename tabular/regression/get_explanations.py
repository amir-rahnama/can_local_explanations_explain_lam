import numpy as np
import pandas as pd
import os 

import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats
from scipy.stats import norm, spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.base import TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import pairwise_distances, accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity, kernel_metrics
import lime
import lime.lime_tabular
import shap
from tabulate import tabulate
import joblib
import pickle
import scipy 
import re


BASE_PATH ='/home/amirh/code/can_explanations_explain_lam/tabular/regression/data'
curr_pred_class = 1
exp_type = ['lime', 'shap', 'lpi', 'gt'] 

logger = logging.getLogger('{}/get_exp_new'.format(BASE_PATH))
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('{}/get_exp_new_v12.log'.format(BASE_PATH))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

sample_size = 5000

def find_feature_idx(feature_rule):
    res = re.findall(r'feature_id_(\d+)', feature_rule)
    
    if len(res) > 0:
        return int(res[0])
    
def transform_lime_exp(exp, features):
    transform_exp = np.zeros(len(features))

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X

def lime_exp(x_train, instance, model, d_info):
    fake_feat_names = [] 
    for i in range(x_train.shape[1]):
        f_name = 'feature_id_{}'.format(i)
        fake_feat_names.append(f_name)
        
    background_size = 1000
    if x_train.shape[0] < background_size:
        background_size = x_train.shape[0]

    explainer = lime.lime_tabular.LimeTabularExplainer(x_train[:background_size], feature_names = fake_feat_names, 
                                                   categorical_features=[], categorical_names=[], verbose=False,
                                                   mode='regression')
    exp_lime = explainer.explain_instance(instance, model.predict, num_features=x_train.shape[1], num_samples=sample_size)
    trans_lime = transform_lime_exp(exp_lime.as_list(), fake_feat_names)
    return trans_lime

def shap_exp(x_train, instance, model, d_info):
    background_size = 1000
    if x_train.shape[0] < background_size:
        background_size = x_train.shape[0]
        
    explainer = shap.KernelExplainer(model.predict, x_train[:background_size])
    shap_values = explainer.shap_values(instance, nsamples=sample_size)
    return shap_values

def lpi_exp(x_train, instance, model, d_info):
    instance = instance.reshape(1, x_train.shape[1])
    importance  = np.zeros(instance.shape[1])
    base_pred = model.predict(instance)

    for i in range(0, instance.shape[1]):    
        all_feat_values = np.unique(x_train[:, i])
        new_instance = np.tile(instance, (len(all_feat_values), 1))
        
        for j in range(new_instance.shape[0]):
            new_instance[j, i] = all_feat_values[j] 

        pred = model.predict(new_instance)
        
        importance[i] = np.mean(pred - base_pred)
    
    return importance

def ground_truth(x_train, instance, model, d_info):
    gt = np.multiply(model['lr'].coef_, instance)
    return gt


data_info = pickle.load(open("{}/data_info_v1.p".format(BASE_PATH), "rb"))


def get_all_exp(): 
    max_threshold = 100
    #datasets = ['wineWhite', 'wineRed', 'puma8fh', 'anacalt', 'heating', 'kin8nm', 'istanbul', 'automobile', 'housing', 'insurance', 'quakes', 'kin8fh', 'wizmir', 'bank8fh', 'anacalt', 'kin8nm', 'kin8nh', 'puma8fm', 'bank8nm', 'bank8nh']
    #datasets = [ 'deltaA', 'treasury', 'kin8nm']
    #datasets = [ 'kin8fh', 'wizmir', 'bank8fh', 'anacalt', 'kin8nm', 'kin8nh', 'puma8fm', 'bank8nm', 'bank8nh']
    datasets = ['kin8fm']
    #datasets = ['automobile']
    #preprocessing = ['standard', 'minmax', 'robust']
    preprocessing = ['standard']
    
    
    
    exp_function = {
        'lime': lime_exp,
        'shap': shap_exp,
        'lpi': lpi_exp,
        'gt': ground_truth
    }
    
    # OLd text
    # v10 5000
    # v11 2000
    # v12 7000
    # v13 1000
    # v14 2000 (new datasets)
    
    # V14 5000
    # V15 7000
    # v16 2000
    # v17 1000
    # v18 500
    
    model = 'lr'
    for data_key in datasets:
        print(data_key)
        #folder_name = 'exp_v14'
        folder_name = 'exp_v14'
        
        for preproc in preprocessing:
            if not os.path.exists('{}/{}/{}/{}'.format(BASE_PATH, data_key, preproc, folder_name)):
                os.makedirs('{}/{}/{}/{}'.format(BASE_PATH, data_key, preproc, folder_name))
            
            x_train = np.load('{}/{}/{}/x_train.npy'.format(BASE_PATH, data_key, preproc),  allow_pickle=True)
            x_test = np.load('{}/{}/{}/x_test.npy'.format(BASE_PATH, data_key, preproc),  allow_pickle=True)
            model_object = joblib.load('{}/{}/{}/{}_v2.joblib'.format(BASE_PATH, data_key, preproc, model))
            total_run = x_test.shape[0] if x_test.shape[0] < max_threshold  else max_threshold
            for exp in list(exp_function.keys()):
                logger.info('{}, {}, {}, {}'.format(data_key, preproc, model, exp))
                all_exp = []
                for idx in range(total_run):
                #for idx in range(1):
                    instance_explained = x_test[idx]

                    all_exp.append(exp_function[exp](x_train, instance_explained, model_object, data_info[data_key]))
                    
                np.save('{}/{}/{}/{}/{}_exp_{}_{}_v1.npy'.format(BASE_PATH, data_key, preproc, folder_name, 
                                                              exp, model, preproc), all_exp)
            
if __name__ == "__main__":
    get_all_exp()
    
    
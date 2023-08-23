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

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) == scipy.sparse.csr.csr_matrix:    
            return X.todense()
        else: 
            return X
        
#import sys
#sys.path.append('./')
#from techniques import get_local_permutation_importance, get_lime_explanations, get_shap_explanations, nbayes_local_explaination, lreg_local_explaination, get_local_permutation_importance_odds

#lreg = None
#gbayes = None

#logger = logging.getLogger('get_exp_all')
#logger.setLevel(logging.DEBUG)
#fh = logging.FileHandler('exp.log')
#fh.setLevel(logging.DEBUG)
#logger.addHandler(fh)
sample_size = 5000
#BASE_PATH = '/tf/tabular/data'

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


def get_lime_explanations(explainer, instance, model_fn, features, explained_class):
    exp_lime = explainer.explain_instance(instance, model_fn, num_features=len(features), labels=(explained_class,), top_labels=2, num_samples=sample_size)
    trans_lime = transform_lime_exp(exp_lime.as_list(), features)
    return trans_lime

def get_shap_explanations(instance, model_fn, median_data):
    shap_explainer = shap.KernelExplainer(model_fn, median_data)
    shap_values = shap_explainer.shap_values(instance, nsamples=sample_size)
    
    return shap_values

def odds_fn(z, model=None):
    epsilon = 1e-3

    result = []
    for i in range(0, z.shape[0]):
        instance = z[i].reshape(1, -1)
        pre_original = model.predict_proba(instance)
        pre_flip = np.flip(pre_original) 
        r = np.log(epsilon + (pre_original + 0.5)/ (pre_flip + 0.5))[0]
        result.append(r)

    return np.array(result)

def lime_exp(instance, x_train, d_info, model_obj, dataset_name, model_name):
    cat_feats = list(d_info[dataset_name]['cat_feature_idx'])
    cat_feat_names = list(d_info[dataset_name]['cat_feature_name'])
    feat_names = list(d_info[dataset_name]['features'])
    
    fake_feat_names = [] 
    fake_cat_feat_names = []
    for i in range(len(feat_names)):
        f_name = 'feature_id_{}'.format(i)
        fake_feat_names.append(f_name)
        if len(cat_feats) > 0 and (i in cat_feats): 
            fake_cat_feat_names.append(f_name)

    explained_class = 1
    
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, class_names=[0, 1], 
                                                                feature_names = fake_feat_names,
                                                                categorical_features=cat_feats,
                                                                categorical_names=fake_cat_feat_names,
                                                                verbose=False)
   
    
    lime_exp = get_lime_explanations(explainer, instance, model_obj.predict_proba, feat_names, explained_class)
    return lime_exp


def shap_exp(instance, x_train, d_info, model_obj, dataset_name, model_name):
    
    odds_fn_lmbda = lambda z: odds_fn(z, model_obj)[:, 1]
    median_train = np.median(x_train, axis=0).reshape(1, -1)
    shap_exp = get_shap_explanations(instance, odds_fn_lmbda, median_train)
    
    return shap_exp


def nbayes_local_explaination(instance, model, explained_class):
    feature_importance = []
    epsilon = 1e-3

    for i in range(instance.shape[0]):
        '''explained_class_prob = norm.pdf(instance[i], model.theta_[ explained_class][i], 
                                        np.sqrt(model.sigma_[explained_class][i]))
        unexplained_class_prob = norm.pdf(instance[i], model.theta_[1 - explained_class][i], 
                                          np.sqrt(model.sigma_[1 - explained_class][i])) + epsilon'''
        explained_class_prob = norm.pdf(instance[i], model.theta_[ explained_class][i], 
                                        np.sqrt(model.var_[explained_class][i]))
        unexplained_class_prob = norm.pdf(instance[i], model.theta_[1 - explained_class][i], 
                                          np.sqrt(model.var_[1 - explained_class][i])) + epsilon
        feature_importance.append(np.log(epsilon + (explained_class_prob / unexplained_class_prob)))
    
    return np.array(feature_importance)


def lreg_local_explaination(instance, model, predicted_class):
    return np.multiply(model.coef_, instance).flatten()

def t_ground_truth(all_col_list, d_info):
    new_index_range = {}
    features = d_info['features']
    
    cat_feature_idx = np.sort(d_info['cat_feature_idx'])
    
    col_names = np.array(features)[cat_feature_idx]
    
    for k in cat_feature_idx:
        new_index_range[k] = []

    for j in range(len(all_col_list)):
        c = all_col_list[j]
        if c.find('_') != -1: 
            res = c.split('_')[0]
            if res in col_names:
                c_idx = np.argwhere(res == np.array(features))[0][0]
                new_index_range[c_idx].append(j)
    return new_index_range

def ground_truth(instance, x_train, d_info, model_obj, dataset_name, model_name):
    has_categories = ('preprocess' == model_obj.steps[0][0])
    instance__ = instance.copy()

    if has_categories:
        #print(model_obj['preprocess'])
        instance = model_obj['preprocess'].transform(instance.reshape(1, -1))
        s = model_obj['to_dense'].transform(instance.reshape(1, -1))

    #print('instance', instance)
    instance_ = np.squeeze(np.asarray(instance))
    

    instance_new = model_obj['to_dense'].transform(instance.reshape(1, -1))
    instance_new = np.squeeze(np.asarray(instance_new))
    #if model_type == sklearn.naive_bayes.GaussianNB:
    
    if model_name == 'gbayes':
        gt_exp = nbayes_local_explaination(instance_new, model_obj[model_name], 1)
    else: 
        #print(model_obj[model_name].coef_.shape, instance_new.shape)
        #print('instance new', instance_new, instance_new.shape, model_obj[model_name].coef_.shape)
        gt_exp = lreg_local_explaination(instance_new, model_obj[model_name], 1)
    #print('gt_exp', gt_exp.shape)
    if has_categories: 
        #print(instance_new.shape)
        #print(x_train.shape)
        #print(instance_new.shape)
        #print(d_info['churn'])
        
        t = model_obj['preprocess'].transformers_[0][1]
        
        cat_map = t_ground_truth(t.get_feature_names(), d_info[dataset_name])
        gt_exp_new = np.zeros(instance__.shape[0])
        c_features = list(cat_map.keys())

        for c in c_features:
            gt_exp_new[c] = np.sum(gt_exp[cat_map[c]])
        all_features = np.arange(x_train.shape[1])
        not_col_idx = np.setxor1d(all_features, c_features).astype(np.int32)
        gt_exp_new[not_col_idx] = gt_exp[not_col_idx]
        
        gt_exp = gt_exp_new
        
    return gt_exp.flatten()
    
    return gt_exp

def get_lpi_explanations(data, instance, model, explained_class):
    instance = instance.reshape(1, data.shape[1])
    importance  = np.zeros(instance.shape[1])
    #base_pred = model.predict_proba(instance)[:, explained_class]
    base_odds = odds_fn(instance, model)[0][explained_class]

    for i in range(0, instance.shape[1]):    
        all_feat_values = np.unique(data[:, i])
        new_instance = np.tile(instance, (len(all_feat_values), 1))
        
        for j in range(new_instance.shape[0]):
            new_instance[j, i] = all_feat_values[j] 

        #pred = model.predict_proba(new_instance)[:, explained_class]
        new_odds = odds_fn(new_instance, model)[:, explained_class]
        importance[i] = np.mean(base_odds - new_odds)
    
    return importance

def lpi_exp(instance, x_train, d_info, model_obj, dataset_name, model_name):
    return get_lpi_explanations(x_train, instance, model_obj, 1)


preprocessing = ['robust', 'standard', 'minmax']

models = ['lreg', 'gbayes']
#BASE_PATH = '/tf/tabular'
BASE_PATH ='/home/amirh/code/local_explanations_whitebox/tabular/data'
curr_pred_class = 1
exp_type = ['lime', 'shap', 'lpi', 'gt'] 

logger = logging.getLogger('{}/get_exp_new'.format(BASE_PATH))
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('{}/get_exp_new_v13.log'.format(BASE_PATH))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

#datasets = ['breast_cancer', 'pima_indians', 'banknote', 'iris', 'haberman', 'spambase', 'heart_disease', 'churn', 'hattrick', 'hr', 'audit', 'loan', 'seismic', 'thera', 'titanic', 'donor', 'adult', 'insurance', 'banking', 'attrition' ]

datasets = ['loan', 'breast_cancer', 'donor']

# churn, seimic, titanic, 'donor'
#datasets = ['churn', 'seismic', 'titanic', 'donor']
data_info = pickle.load(open("{}/data_info_v2.p".format(BASE_PATH), "rb"))

def get_all_exp():             
    
    max_threshold = 100
    exp_function = {
        'lime': lime_exp,
        'shap': shap_exp,
        #'lpi': lpi_exp,
        #'gt': ground_truth
    }
    
    # v10 5000
    # v12 7000
    for data_key in datasets:
        folder_name = 'exp_v13'
        
        for preproc in preprocessing:
            if not os.path.exists('{}/{}/{}/{}'.format(BASE_PATH, data_key, preproc, folder_name)):
                os.makedirs('{}/{}/{}/{}'.format(BASE_PATH, data_key, preproc, folder_name))
            
            x_test = np.load('{}/{}/{}/x_test.npy'.format(BASE_PATH, data_key, preproc),  allow_pickle=True)
            total_run = x_test.shape[0] if x_test.shape[0] < max_threshold  else max_threshold
            
            x_train = np.load('{}/{}/{}/x_train.npy'.format(BASE_PATH, data_key, preproc),  allow_pickle=True)
                        
            for model in models:
                model_object = joblib.load('{}/{}/{}/{}_v1.joblib'.format(BASE_PATH, data_key, preproc, model))

                for exp in list(exp_function.keys()):
                    logger.info('{}, {}, {}, {}'.format(data_key, preproc, model, exp))
                    #print('{}, {}, {}, {}'.format(data_key, preproc, model, exp))
                    all_exp = []
                    for idx in range(total_run):
                        print('Getting explanations for instance: ', idx)
                        instance_explained = x_test[idx]
                        
                        all_exp.append(exp_function[exp](instance_explained, x_train, data_info, model_object, data_key,  model))
                    
                    np.save('{}/{}/{}/{}/{}_exp_{}_{}.npy'.format(BASE_PATH, data_key, preproc, folder_name, 
                                                                  exp, model, preproc), all_exp)
            
if __name__ == "__main__":
    get_all_exp()
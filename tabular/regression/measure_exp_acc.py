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

#datasets = ['housing', 'automobile', 'insurance']

#preprocessing = ['robust', 'standard', 'minmax']
preprocessing = ['standard']
models = ['lr']
styles = {'odds': ['lime', 'shap', 'permute'], 'proba': ['lime', 'shap', 'permute']}
BASE_PATH ='/home/amirh/code/can_explanations_explain_lam/tabular/regression/data'



def get_measure(folder_name, exp_models, datasets):
    measure = {}
    for preproc in preprocessing:
        measure[preproc] = {}
        

        for model in models:
            measure[preproc][model] = {}
            for exp_model in exp_models:
                measure[preproc][model][exp_model] = []
                print(exp_model) 
                for dataset in datasets:
                    print(dataset)
                    local_exp = np.load('{}/{}/{}/{}/gt_exp_{}_{}_v1.npy'.format(BASE_PATH, dataset, preproc, folder_name,
                                                                      model, preproc), allow_pickle=True)
                    print(local_exp.shape)
                    x_test = np.load('{}/{}/{}/x_test.npy'.format(BASE_PATH, dataset, preproc),  allow_pickle=True)
                    exp_result = np.load('{}/{}/{}/{}/{}_exp_{}_{}_v1.npy'.format(BASE_PATH, dataset, preproc, folder_name,
                                                                      exp_model, model, preproc), allow_pickle=True)
                    print(exp_result.shape)
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
                                              local_exp[i].reshape(1, -1).astype(np.float64))
                        measure_vals['cosine'].append(cd)


                    measure[preproc][model][exp_model].append(measure_vals)
    return measure

if __name__ == '__main__':
    #folder_name = 'exp_v14'
    #exp_models = [ 'lime', 'shap', 'lpi']
    #datasets = [ 'wineWhite', 'wineRed', 'quakes', 'deltaE', 'deltaA', 'anacalt', 'kin8nm', 'istanbul', 'mortage', 'treasury',  'mg', 'kin8fh']
    #measure = get_measure(folder_name, exp_models, datasets)
    #pickle.dump( measure, open( "{}/exp_reg_accuracy_v16.p".format(BASE_PATH), "wb" ) )
    all_measures = []
    #folder_names = ['exp_v18', 'exp_v17', 'exp_v16', 'exp_v15','exp_v14']
    #datasets = ['deltaA', 'treasury', 'kin8nm']
    exp_models = [ 'lime', 'shap', 'lpi']
    datasets =['wineWhite', 'wineRed', 'puma8fh', 'anacalt',  'kin8nm', 'istanbul', 'bank8fm', 'mg', 'treasury', 'kin8fm',
               'quakes', 'kin8fh', 'wizmir', 'bank8fh',  'kin8nh', 'puma8fm', 'bank8nm', 'bank8nh', 'deltaA', 'deltaE', 'mortage']
    #for f_name in folder_names: 
    #   measure = get_measure(f_name, exp_models, datasets)
    #    all_measures.append(measure)
    f_name = 'exp_v14'
    measure = get_measure(f_name, exp_models, datasets)
    pickle.dump( measure, open( "{}/exp_reg_accuracy_v17.p".format(BASE_PATH), "wb" ) )
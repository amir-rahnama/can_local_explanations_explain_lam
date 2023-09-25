# Can Local Additive Explanations Explain Linear Additive Models? 

This repository contains the code for our paper, "Can Local Additive Explanations Explain Linear Additive Models?" ECML PKDD 2023 (Journal Track). In our evaluation, we measure the local accuracy of LIME, KernelSHAP, and Local Permutation Importance (LPI) explanations when explaining linear additive models such as Linear and Logistic Regression and Naive Bayes models. Our study shows that the explanations of LIME and KernelSHAP are not accurate for linear classification models of Logistic Regression and Naive Bayes. The full paper can be accessed here: https://link.springer.com/article/10.1007/s10618-023-00971-3. 

The presentation of our paper in ECML PKDD 2023 is accessible on YouTube: https://youtu.be/np1UjSTGs_I?si=0bOOiRmDdGZlEw14. 

A more straightforward summary of our work that is more welcoming to people who are not experts in Explainable Machine Learning (XAI) is available at URL.

# Prerequisits 

Our experiments require you to have the following Python packages installed: 

* ScikitLearn version 1 
* Numpy, Pandas and Scipy
* Tabulate 
* Seneca (for running the Motivating example only): https://github.com/riccotti/SyntheticExplanationGenerator.


# How to run the experiments. 

In general, our experiments need to run in order: 

1. Make sure that the datasets are in place. Download the data files from Google Drive (https://drive.google.com/drive/folders/1W3up3L6YI9E8kPIL4jriX16BP0RosP-Z?usp=sharing) and place them under /tabular/data and tabular/regression/data folders.
2. Train the models on the datasets. Logistic Regression and Naive Bayes are trained via tabular/generate_data_train_models.py and Linear Regression via /tabular/regression/trail_all_models.py
3. Get local explanations and ground truth importance scores. For Logistic Regression and Naive Bayes, run tabular/get_all_explanations.py, and for Linear Regression, run /tabular/regression/get_explanations.py
4. Measure explanation accuracy. For Logistic Regression and Naive Bayes, run tabular/measure_exp_accuracy.py, and for Linear Regression, run /tabular/regression/measure_exp_acc.py

We have included a Jupyternotebook with the analysis of all our results in classification in tabular/Visual Analysis.ipynb and tabular/regression/Regression Data Analysis.ipynb. 


# Factors
In Sections 5.2.2 and 5.2.3 of our paper, we analyze factors that can affect the local accuracy of explanations. These experiments can be found under /factors for the synthetic use cases. For tabular datasets, these analyses are included in tabular/Visual Analysis.ipynb for classification models and tabular/regression/Regression Data Analysis.ipynb for regression models.

# Robustness

In Section 5.2.8 of our paper, we measure the robustness of local explanations for our classification by running /tabular/measure_explanations_robustness_v2.py and for regression models by running /tabular/regression/get_robustness_regression.py. 

The analysis for the robustness of local explanation in classification models can be found at /tabular/Robustness.ipynb and for regression models in tabular/regression/Regression Data Analysis.ipynb. 

# Motivating Example 

For running our motivating example in Sectio 3, you need to clone the code for Seneca downloaded and included in your Python path:  https://github.com/riccotti/SyntheticExplanationGenerator. After that you can replicate the examples by running /motivating/Motivating Example.ipynb



 

  

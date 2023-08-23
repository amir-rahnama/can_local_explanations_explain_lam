
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
base_path =os.getcwd()

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
    #base_path = '/tf/tabular'
    data = {}
    anchor_dict = {}
    # 1. Breast Cancer
    breast_cancer = load_breast_cancer()
    features = np.array(breast_cancer.feature_names)
    data['breast_cancer'] = [breast_cancer.data, breast_cancer.target, features, [], []]

    # 2. Diabetes
    diabetes = pd.read_csv('{}/pima_indians/diabetes.csv'.format(base_path))
    features = np.array(diabetes.columns)[:-1]
    y = diabetes.iloc[:, -1]
    X = diabetes.iloc[:, 0:-1]
    data['pima_indians'] = [X.to_numpy(), y.values, features, [], []]


    #3. Bank Notes
    banknote = pd.read_csv('{}/banknote/data_banknote_authentication.txt'.format(base_path))
    features = np.array(['variance', 'skewness', 'curtosis', 'entropy'])
    banknote.columns = np.concatenate((features, ['class']))
    y = banknote['class'].to_numpy()
    X = banknote.drop(['class'], axis=1).to_numpy()
    data['banknote'] = [X, y, features, [], []]

    # 4. Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    filterd_values = np.logical_or(y==0, y==1)
    X = X[filterd_values]
    y = y[filterd_values]
    features = np.array(['sepal_length', 'sepal_width', 'petal_length', 'petal width'])
    data['iris'] = [X, y, features, [], []]

    # 5. Haberman
    haberman = pd.read_csv('{}/haberman/haberman.data'.format(base_path))
    features = np.array(['Age','Operation_Year','Positive_lymph_node'])
    haberman.columns = np.concatenate((features, ['survival_status']))
    y = haberman['survival_status'].to_numpy()
    y[y==2] = 0
    haberman = haberman.drop(['survival_status'], axis=1).to_numpy()
    data['haberman'] = [haberman, y, features, [], []]

    #6. Spambase
    spambase = pd.read_csv( '{}/spambase/train_data.csv'.format(base_path))
    labels = spambase['ham'].astype('uint8')
    spambase = spambase.drop(columns=['ham'])
    #features = np.array(spambase.columns)
    features = np.array(['word_freq_make', 'word_freq_address', 'word_freq_all',
           'word_freq_threed', 'word_freq_our', 'word_freq_over',
           'word_freq_remove', 'word_freq_internet', 'word_freq_order',
           'word_freq_mail', 'word_freq_receive', 'word_freq_will',
           'word_freq_people', 'word_freq_report', 'word_freq_addresses',
           'word_freq_free', 'word_freq_business', 'word_freq_email',
           'word_freq_you', 'word_freq_credit', 'word_freq_your',
           'word_freq_font', 'word_freq_zerozerozero', 'word_freq_money',
           'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
           'word_freq_sixfifty', 'word_freq_lab', 'word_freq_labs',
           'word_freq_telnet', 'word_freq_eightfiveseven', 'word_freq_data',
           'word_freq_fourfifteen', 'word_freq_eightfive', 'word_freq_technology',
           'word_freq_nineteennintynine', 'word_freq_parts', 'word_freq_pm',
           'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
           'word_freq_original', 'word_freq_project', 'word_freq_re',
           'word_freq_edu', 'word_freq_table', 'word_freq_conference',
           'char_freq_semicolor', 'char_freq_parant', 'char_freq_bracket', 'char_freq_exclamation',
           'char_freq_dollar_sign', 'char_freq_hashtag', 'capital_run_length_average',
           'capital_run_length_longest', 'capital_run_length_total', 'Id'])

    data['spambase'] = [spambase.to_numpy(), labels, features, [], []]
    
    # 7. Adult
    #adult, y = shap.datasets.adult()
    #y = y.astype(int)
    #features =np.array(['Age', 'Workclass', 'Education Num', 'Marital Status',
    #       'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain',
    #       'Capital Loss', 'Hours per week', 'Country'])
    #target_names = ['<=50K', '>50K']
    #categorical_columns= ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    
    adult = pd.read_csv('{}/adult/adult.data'.format(base_path), header=None)
    labels = adult.iloc[:, 14].values
    adult = adult.drop(columns=[14])
    
    categorical_columns= ['education', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
            'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
           'capital-loss', 'hours-per-week', 'native-country']
    
    adult.columns = features
    
    y = np.zeros(labels.shape[0])
    one_idx = np.argwhere(labels == ' >50K').flatten()
    y[one_idx] = 1
    
    anchor_dict['adult'] = get_unique_cat(adult, categorical_columns)
    
    data['adult'] = [adult, y, features, categorical_columns]

    # 8. Titanic
    titanic = pd.read_csv('{}/titanic/data/train.csv'.format(base_path))
    titanic = titanic.drop(['Cabin', 'Ticket','Name'], axis=1)
    sex = pd.get_dummies(titanic['Sex'],drop_first=True)
    embarked = pd.get_dummies(titanic['Embarked'], drop_first=True)
    titanic.drop(['Sex','Embarked'], axis=1, inplace=True)
    titanic=pd.concat([titanic,sex,embarked], axis=1)
    titanic = titanic.fillna(titanic.mean())
    y = titanic['Survived']
    titanic = titanic.drop(['Survived', 'PassengerId'], axis=1)
    features = titanic.columns.to_numpy()
    categorical_columns = ['S', 'Pclass']
    data['titanic'] = [titanic, y, features, categorical_columns]
    

    #9. Heart Disease
    heart = pd.read_csv('{}/heart_disease/data/heart.csv'.format(base_path))
    y = heart['target']
    heart = heart.drop(['target'], axis=1)
    features = heart.columns.to_numpy()
    categorical_columns = ['sex', 'cp','restecg', 'fbs', 'exang','slope', 'thal']
    data['heart_disease'] = [heart, y, features, categorical_columns]

    # 10. Churn
    churn = pd.read_csv('{}/churn/Churn_Modelling.csv'.format(base_path))
    y = churn['Exited']
    churn = churn.drop(['RowNumber', 'Surname', 'CustomerId', 'Exited'], axis=1)
    categorical_columns = ['Geography', 'Gender']
    features = np.array(churn.columns)
    
    
    #enc = LabelEncoder()
    #churn['Gender'] = enc.fit_transform(churn['Gender'])
    #churn['Geography'] = enc.fit_transform(churn['Geography'])
    
    data['churn'] = [churn, y, features, categorical_columns]

    # 11. hattrick

    HT = pd.read_csv('{}/hattrick/Datos_HT.csv'.format(base_path), sep=';')
    HT = HT[HT['Home midfield']!=0 & (HT['Home midfield']!=0)]
    HT = HT[~np.isnan(HT['Home midfield'])]
    HT['Home Winner'] = HT['Home Goals']>HT['Away Goals']
    HT = HT.drop(['MatchId','Home Attitude', 'Away Attitude'], axis=1)
    categorical_columns = ['Home Tactic', 'Home Tactic Level', 'Away Tactic Level', 'Away Tactic']

    #enc = LabelEncoder()
    #HT['Home Tactic'] = enc.fit_transform(HT['Home Tactic'])
    #HT['Home Tactic Level'] = enc.fit_transform(HT['Home Tactic Level'])
    #HT['Away Tactic Level'] = enc.fit_transform(HT['Away Tactic Level'])
    #HT['Away Tactic'] = enc.fit_transform(HT['Away Tactic'])
    HT['Home Winner'] = HT['Home Winner'].astype(int)

    y = HT['Home Winner']
    HT = HT.drop(['Home Winner'], axis=1)
    features = np.array(HT.columns)

    data['hattrick'] = [HT, y, features, categorical_columns]

    #12. HR 
    hr = pd.read_csv('{}/hr/aug_train.csv'.format(base_path))
    hr = hr.drop(['enrollee_id'], axis=1)

    categorical_columns = ['gender', 'city', 'relevent_experience', 
                           'enrolled_university', 'education_level', 'major_discipline', 
                           'experience', 'company_size', 'company_type','last_new_job']

    #enc = LabelEncoder()
    #for cat in categorical_columns:
    #    hr[cat] = enc.fit_transform(hr[cat])

    #hr['gender'] = hr['gender'].astype(int)
    #categorical_columns.append('gender')

    y = hr['target'].astype(int)
    hr = hr.drop(['target'], axis=1)
    features = np.array(hr.columns)
    data['hr'] = [hr, y, features, categorical_columns]

    # 13. Insurance 
    insurance = pd.read_csv('{}/insurance/travel insurance.csv'.format(base_path))
    y = insurance['Claim']
    insurance = insurance.drop(['Claim'], axis=1)

    insurance.columns = ['Agency', 'Type', 'Distribution', 'Product',
                       'Duration', 'Destination', 'Sales', 'Commision',
                       'Gender', 'Age']

    numerical_columns = ['Sales', 'Commision', 'Age', 'Duration']
    categorical_columns = np.setxor1d(insurance.columns, numerical_columns)

    #enc = LabelEncoder()

    #for cat in categorical_columns:
    #    insurance[cat] = enc.fit_transform(insurance[cat])

    y[y.values == 'No'] = 0
    y[y.values == 'Yes'] = 1
    y = y.astype(int)
    features = np.array(insurance.columns)

    data['insurance'] = [insurance, y, features, categorical_columns.tolist()]

    # 14. Audit

    audit = pd.read_csv('{}/audit/audit_data.csv'.format(base_path))
    y = audit['Risk'].values
    audit = audit.drop(['Risk'], axis=1)
    categorical_columns = ['LOCATION_ID', 'District_Loss', 'History']
    #enc = LabelEncoder()

    #for cat in categorical_columns:
    #    audit[cat] = enc.fit_transform(audit[cat])
    audit['Money_Value'] = audit['Money_Value'].fillna(audit['Money_Value'].mean())

    features = np.array(audit.columns)
    data['audit'] = [audit, y, features, categorical_columns]

    # 15. Loan
    loan = pd.read_csv('{}/loan/Bank_Personal_Loan_Modelling-1.xlsx'.format(base_path))
    y = loan['Personal Loan']
    loan = loan.drop(['ID'], axis=1)
    categorical_columns = ['ZIP Code', 'Family','Education', 'Mortgage', 'CD Account', 'Online', 'Securities Account', 
                           'CreditCard']
    #enc = LabelEncoder()

    #for cat in categorical_columns:
    #    loan[cat] = enc.fit_transform(loan[cat])
    features = np.array(loan.columns)
    data['loan'] = [loan, y, features, categorical_columns]

    #16. Attrition

    attrition = pd.read_csv('{}/attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv'.format(base_path))
    attrition = attrition.drop(['EmployeeNumber', 'EmployeeCount'], axis=1)
    y = attrition['Attrition'].values
    y[y == 'Yes'] = 1
    y[y == 'No'] = 0
    y = y.astype(int)

    attrition.columns = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
           'DistanceFromHome', 'Education', 'EducationField',
           'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
           'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
           'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverEighteen',
           'OverTime', 'PercentSalaryHike', 'PerformanceRating',
           'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
           'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
           'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
           'YearsWithCurrManager']

    categorical_columns = ['BusinessTravel', 'Department','Education', 'EnvironmentSatisfaction', 'JobLevel', 
                           'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 
                           'WorkLifeBalance', 'StockOptionLevel', 'EducationField', 'Gender', 'JobRole', 
                           'MaritalStatus', 'OverEighteen', 'OverTime']
    #enc = LabelEncoder()
    #for cat in categorical_columns:
    #    attrition[cat] = enc.fit_transform(attrition[cat])
    features = np.array(attrition.columns)
    data['attrition'] = [attrition, y, features, categorical_columns]


    # 17.  Donors
    donor=pd.read_csv("{}/donor/Raw_Data_for_train_test.csv".format(base_path))
    conditions = [
        (donor['TARGET_D'] >= 50),
        (donor['TARGET_D'] >= 20) & (donor['TARGET_D'] < 50),
        (donor['TARGET_D'] >= 13) & (donor['TARGET_D'] < 20),
        (donor['TARGET_D'] >= 10) & (donor['TARGET_D'] < 13),
        (donor['TARGET_D'] < 10)
        ]
    values = ['A', 'B', 'C', 'D','E']
    donor['DONATION_TYPE'] = np.select(conditions, values)
    donor=donor.drop(['TARGET_D'],axis=1)
    donor = donor.drop(['PUBLISHED_PHONE'],axis=1)
    donor['DONOR_AGE']=donor['DONOR_AGE'].fillna(donor['DONOR_AGE'].mean())
    donor['DONOR_AGE']=donor['DONOR_AGE'].astype('int64')
    donor['INCOME_GROUP']=donor['INCOME_GROUP'].fillna(donor['INCOME_GROUP'].mode()[0])
    donor['INCOME_GROUP']=donor['INCOME_GROUP'].astype('int64')
    donor['WEALTH_RATING']=donor['WEALTH_RATING'].fillna(donor['WEALTH_RATING'].mode()[0])
    donor['WEALTH_RATING']=donor['WEALTH_RATING'].astype('int64')
    donor=donor.dropna()
    donor['SES']=donor['SES'].str.replace('?','2')
    donor['SES']=donor['SES'].astype('int64')
    donor['URBANICITY']=donor['URBANICITY'].str.replace('?','S')
    donor['CLUSTER_CODE']=donor['CLUSTER_CODE'].str.replace('.','40')
    donor['CLUSTER_CODE']=donor['CLUSTER_CODE'].astype('int64')
    y = donor['TARGET_B']
    donor = donor.drop(['TARGET_B'], axis=1)

    donor.columns = ['CONTROL_NUMBER', 'MONTHS_SINCE_ORIGIN', 'DONOR_AGE', 'IN_HOUSE',
           'URBANICITY', 'SES', 'CLUSTER_CODE', 'HOME_OWNER', 'DONOR_GENDER',
           'INCOME_GROUP', 'OVERLAY_SOURCE', 'MOR_HIT_RATE', 'WEALTH_RATING',
           'MEDIAN_HOME_VALUE', 'MEDIAN_HOUSEHOLD_INCOME',
           'PCT_OWNER_OCCUPIED', 'PER_CAPITA_INCOME', 'PCT_ATTRIBUTEONE',
           'PCT_ATTRIBUTE_TWO', 'PCT_ATTRIBUTE_THREE', 'PCT_ATTRIBUTE_FOUR', 'PEP_STAR',
           'RECENT_STAR_STATUS', 'RECENCY_STATUS_NINESIXNK',
           'FREQUENCY_STATUS_NINESEVENNK', 'RECENT_RESPONSE_PROP',
           'RECENT_AVG_GIFT_AMT', 'RECENT_CARD_RESPONSE_PROP',
           'RECENT_AVG_CARD_GIFT_AMT', 'RECENT_RESPONSE_COUNT',
           'RECENT_CARD_RESPONSE_COUNT', 'MONTHS_SINCE_LAST_PROM_RESP',
           'LIFETIME_CARD_PROM', 'LIFETIME_PROM', 'LIFETIME_GIFT_AMOUNT',
           'LIFETIME_GIFT_COUNT', 'LIFETIME_AVG_GIFT_AMT',
           'LIFETIME_GIFT_RANGE', 'LIFETIME_MAX_GIFT_AMT',
           'LIFETIME_MIN_GIFT_AMT', 'LAST_GIFT_AMT', 'CARD_PROM_ONETWO',
           'NUMBER_PROM_ONETWO', 'MONTHS_SINCE_LAST_GIFT',
           'MONTHS_SINCE_FIRST_GIFT', 'FILE_AVG_GIFT', 'FILE_CARD_GIFT',
           'DONATION_TYPE']
    
    categorical_columns = ['URBANICITY', 'HOME_OWNER', 'DONOR_GENDER', 
                           'OVERLAY_SOURCE', 'RECENCY_STATUS_NINESIXNK', 'DONATION_TYPE']
    #enc = LabelEncoder()

    #for cat in categorical_columns:
    #    donor[cat] = enc.fit_transform(donor[cat])
    features = np.array(donor.columns)

    data['donor'] = [donor, y, features, categorical_columns]

    # 18. Seismic
    seismic = pd.read_csv("{}/seismic/seismic-bumps.csv".format(base_path))
    y = seismic['class']
    seismic = seismic.drop(['class'], axis=1)
    seismic.columns = ['seismic', 'seismoacoustic', 'shift', 'genergy', 'gpuls',
           'gdenergy', 'gdpuls', 'hazard', 'nbumps', 'nbumpstwo', 'nbumpsthree',
           'nbumpsfour', 'nbumpsfive', 'nbumpssix', 'nbumpsseven', 'nbumpseightone', 'energy',
           'maxenergy']
    categorical_columns = ['seismic', 'seismoacoustic', 'shift', 'hazard']
    #enc = LabelEncoder()
    #for cat in categorical_columns:
    #    seismic[cat] = enc.fit_transform(seismic[cat])
    features = np.array(seismic.columns)
    data['seismic'] = [seismic, y, features, categorical_columns]

    # 19. Thera Personal Loan
    thera = pd.read_csv('{}/thera/Bank_Personal_Loan_Modelling.csv'.format(base_path))
    y = thera['Personal Loan']
    thera = thera.drop(['ID', 'ZIP Code'], axis = 1)
    categorical_columns = ['Education', 'Mortgage']
    
    #enc = LabelEncoder()
    #for cat in categorical_columns:
    #    thera[cat] = enc.fit_transform(thera[cat])
    features = np.array(thera.columns)
    data['thera'] = [thera, y, features, categorical_columns]

    # 20. Banking
    banking = pd.read_csv('{}/banking/new_train.csv'.format(base_path))
    y = banking['y'].values
    y[y == 'yes'] = 1
    y[y == 'no'] = 0
    y = y.astype(int)
    banking = banking.drop(['y'], axis=1)
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
           'month', 'day_of_week', 'poutcome']
    #enc = LabelEncoder()
    #for cat in categorical_columns:
    #    banking[cat] = enc.fit_transform(banking[cat])
    features = np.array(banking.columns)
    data['banking'] = [banking, y, features, categorical_columns]

    return data

def generate_data_info(): 
    data = get_all_data()
    d_info = {}
    
    for data_key in data:
        print(data_key)
        cat_col_idx = get_cat_col_idx(data[data_key][0], data[data_key][3])
        d_info[data_key] = {'features': data[data_key][2], 'cat_feature_name': data[data_key][3], 'cat_feature_idx': cat_col_idx}
        
    pickle.dump(d_info, open("/tf/tabular/data_info.p", "wb"))
    
    
def generate_data():
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    categorical_names = {}
    preproc = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}
    data = get_all_data()

    all_keys = ['breast_cancer', 'hattrick', 'pima_indians', 'banknote', 'iris', 
                'haberman', 'spambase', 'titanic', 'heart_disease', 'churn', 
                 'hr', 'audit', 'loan', 'attrition', 
                'donor', 'seismic', 'thera', 'adult', 'insurance', 'banking']                  

    for data_key in all_keys:
        print('dataset:', data_key)
        categorical_names[data_key] = {}
        
        new_data = data[data_key]
        data_cat_cols = new_data[3]
        for pre_ in preproc.keys():
            print('preprocessing ', pre_)

            if not os.path.exists('{}/{}/{}'.format(base_path, data_key, pre_)):
                os.makedirs('{}/{}/{}'.format(base_path, data_key, pre_))
                
            if not isinstance(new_data[0], pd.DataFrame):
                new_data[0] = pd.DataFrame(new_data[0])

            cat_col_idx = get_cat_col_idx(new_data[0], new_data[3])
            all_features = np.arange(new_data[0].shape[1])
            not_col_idx = np.setxor1d(all_features, cat_col_idx).astype(np.int32)
            data_col_temp = new_data[0].iloc[:, not_col_idx]
            transformed_result = preproc[pre_].fit_transform(data_col_temp)
            new_data[0].iloc[:, not_col_idx] = transformed_result
            print(new_data[0])
            if len(cat_col_idx) > 0: 
                for cat in list(data_cat_cols):
                    
                    enc = LabelEncoder()
                    new_data[0][cat] = enc.fit_transform(new_data[0][cat])
                    categorical_names[data_key][cat] = enc.classes_.tolist()
            print(new_data[0])
            sys.exit()
            X_train, X_test, y_train, y_test = train_test_split(new_data[0], new_data[1], random_state=0)

            np.save('{}/{}/{}/x_test.npy'.format(base_path, data_key, pre_), X_test)
            np.save('{}/{}/{}/x_train.npy'.format(base_path, data_key, pre_), X_train)

            if len(cat_col_idx) > 0:
                categorical_encoder_gb = OneHotEncoder(handle_unknown="ignore")    
                preprocessing_gb = ColumnTransformer([
                    ("cat", categorical_encoder_gb, cat_col_idx)
                ])
                gbayes = Pipeline([
                    ("preprocess", preprocessing_gb),
                    ('to_dense', DenseTransformer()), 
                    ("gbayes", GaussianNB())
                ])
                
                categorical_encoder_lreg = OneHotEncoder(handle_unknown="ignore")
                preprocessing_lreg = ColumnTransformer([
                    ("cat", categorical_encoder_lreg, cat_col_idx)
                ])
                lreg = Pipeline([
                    ("preprocess", preprocessing_lreg),
                    ('to_dense', DenseTransformer()), 
                    ("lreg", LogisticRegression(random_state=0, max_iter=10000))
                ])
            else: 
                gbayes = Pipeline([
                    ('to_dense', DenseTransformer()), 
                    ("gbayes", GaussianNB())
                ])
                
                lreg = Pipeline([
                    ('to_dense', DenseTransformer()), 
                    ("lreg", LogisticRegression(random_state=0, max_iter=10000))
                ])
            

            gbayes.fit(X_train, y_train)
            lreg.fit(X_train, y_train)
            
            gbayes = GaussianNB().fit(X_train, y_train)
            lreg = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
            
            gbayes_obj = dump(gbayes, '/tf/tabular/{}/{}/gbayes_v1.joblib'.format(data_key, pre_))
            lreg_obj = dump(lreg, '/tf/tabular/{}/{}/lreg_v1.joblib'.format(data_key, pre_))
            
    pickle.dump(categorical_names, open("/tf/tabular/lime_dict.p", "wb"))
if __name__ == "__main__":
    generate_data()
    generate_data_info()

from sklearn.datasets import make_classification
import pandas as pd
from diro2c.data_generation.helper import replace_by_most_used_value, replace_by_median,recognize_features_type
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import defaultdict
from diro2c.data_generation.compare_functions import *

def loaddata(name):
    if name == "running1":
        np.random.seed(1)
        x1 = np.random.multivariate_normal((-5, 0), [[4, 0], [0, 4]], 500)
        x2 = np.random.multivariate_normal((5, 0), [[4, 0], [0, 4]], 500)
        #x3 = np.random.multivariate_normal((12.5, 0), [[7, 0], [0, 7]], 333)
        #data = pd.DataFrame(np.concatenate((x1,x2,x3)), columns=['x1', 'x2'])
        data = pd.DataFrame(np.concatenate((x1,x2)), columns=['x1', 'x2'])
        return [data, ['x1', 'x2']]
    if name == "running1test":
        np.random.seed(2)
        x1 = np.random.multivariate_normal((-5, 0), [[4, 0], [0, 4]], 250)
        x2 = np.random.multivariate_normal((5, 0), [[4, 0], [0, 4]], 250)
        data = pd.DataFrame(np.concatenate((x1,x2)), columns=['x1', 'x2'])
        return [data, ['x1', 'x2']]
    if name == "running2":
        np.random.seed(1)
        #https://cs231n.github.io/neural-networks-case-study/
        N = 333
        D = 2 # dimensionality
        K = 3 # number of classes
        data = np.zeros((N*K,D))
        labels = np.zeros(N*K, dtype='uint8')
        for j in range(K):
            ix = range(N*j,N*(j+1))
            r = np.linspace(0.0,1,N) # radius
            t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.5 # theta = 0.5
            data[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            labels[ix] = j
        data = pd.DataFrame(data, columns = ['x1', 'x2'])
        #data['y'] = labels
        return [data, ['x1', 'x2']]
    if name == "running2test":
        np.random.seed(2)
        #https://cs231n.github.io/neural-networks-case-study/
        N = 500
        D = 2 # dimensionality
        K = 3 # number of classes
        data = np.zeros((N*K,D))
        labels = np.zeros(N*K, dtype='uint8')
        for j in range(K):
            ix = range(N*j,N*(j+1))
            r = np.linspace(0.0,1,N) # radius
            t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.5 # theta = 0.5
            data[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            labels[ix] = j
        data = pd.DataFrame(data, columns = ['x1', 'x2'])
        data['y'] = labels
        return [data, ['x1', 'x2']]
    if name == 'bankmarketing':
        dataA = pd.read_csv('data/bank-additional/bank-additional-full.csv', index_col=False, sep=';')
        #Determine type of variable:
        types = dataA.dtypes
        categorical = dataA.columns[types == 'object'].drop('education')
        numeric = dataA.columns[types != 'object']
        #Label encoding:
        d = defaultdict(LabelEncoder)
        dataA[categorical] = dataA[categorical].apply(lambda x: d[x.name].fit_transform(x))
        #ordinal variables:
        dataA['education'] = dataA['education'].apply(lambda x: ["unknown", "basic.4y", "basic.6y", "basic.9y",
                                                                 "high.school","illiterate","professional.course",
                                                                 "university.degree"].index(x))

        entrepreneur = d['job'].transform(['entrepreneur'])[0]
        management = d['job'].transform(['management'])[0]
        unknowndegree = 0
        basic4degree = 1

        dataB = dataA.copy()
        dataB.loc[:,'pdays'] = dataB.loc[:,'pdays'] + 5
        ind = ((dataB.job == entrepreneur) | (dataB.job == management)) & \
            ((dataB.education != unknowndegree) | (dataB.education != basic4degree))
        dataB.loc[ind,'education'] = dataB.loc[ind, 'education'] - 1

        dataB.loc[dataB.housing>0, 'age'] = dataB.loc[dataB.housing>0, 'age'] + 10
        dataB.loc[dataB.loan>0, 'age'] = dataB.loc[dataB.loan>0, 'age'] - 10

        return [dataA, dataB, ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                               'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                               'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                               'cons.conf.idx', 'euribor3m', 'nr.employed'],
                ['job', 'marital', 'education', 'default', 'housing', 'loan',
                 'contact', 'month', 'day_of_week', 'poutcome'],
                ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
                 'cons.conf.idx', 'euribor3m', 'nr.employed'],
                d]
    if name == 'compas':
        dataA = pd.read_csv('https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv', sep=',')
        dataA = dataA[['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                   'priors_count', 'c_charge_degree', 'score_text']]
        #Determine type of variable:
        types = dataA.dtypes
        categorical = dataA.columns[types == 'object'].drop('score_text')
        numeric = dataA.columns[types != 'object']
        #Label encoding:
        d = defaultdict(LabelEncoder)
        dataA[categorical] = dataA[categorical].apply(lambda x: d[x.name].fit_transform(x))
        africanamerican = d['race'].transform(['African-American'])[0]
        chargedegreem = d['c_charge_degree'].transform(['M'])[0]
        #ordinal features:
        dataA['score_text'] = dataA['score_text'].apply(lambda x: ['Low', 'Medium', 'High'].index(x))
        dataA.rename(columns = {'score_text':'y'}, inplace=True)

        dataB = dataA.copy()
        dataB.loc[dataB.age > 60,'priors_count'] = dataB.loc[dataB.age > 60, 'priors_count'] + 5
        dataB.loc[(dataB.age < 30) & (dataB.priors_count<5),'priors_count'] = 0
        dataB.loc[(dataB.age < 30) & (dataB.priors_count>=5),'priors_count'] = dataB.loc[(dataB.age < 30) & (dataB.priors_count>=5), 'priors_count'] - 5
        dataB.loc[(dataB.race == africanamerican) & (dataB.juv_fel_count <=2),'c_charge_degree'] = chargedegreem

    return [dataA, dataB, ['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count',
                               'juv_other_count', 'priors_count', 'c_charge_degree'],
            ['sex', 'race', 'c_charge_degree'], ['age','juv_fel_count', 'juv_misd_count',
                                                 'juv_other_count', 'priors_count'], d]

#adapted from diro2c
def prepare_df(df, datasetname, class_name, discrete, continuous, replacing_mv = True):
    """
    :param df: label-encoded pandas dataframe
    :param datasetname: string, name of the dataset
    :param class_name: string, name of outcome
    :param discrete: list of strings, names of discrete columns
    :param continuous: list of strings, names of continuous columns
    :param replacing_mv: boolean, default True. If True, replace missing values by median value
    :return: a dictionary ...
    """

    # set all columns of dataframe
    columns = df.columns.tolist()

    columns_for_decision_rules = list(columns)
    columns_for_decision_rules.remove(class_name)

    # 1. Step: set type for each feature/column of dataset
    type_features, features_type = recognize_features_type(df)

    # 2. Step: replace missing values of continuous and discrete features
    if replacing_mv:
        df = replace_by_median(df, continuous)
        df = replace_by_most_used_value(df, discrete)

    tmp_columns = list(columns)
    tmp_columns.remove(class_name)

    possible_outcomes = list(df[class_name].unique())

    # set index for features
    idx_features = {i: col for i, col in enumerate(
        list(tmp_columns))}

    # 4. Step: extract features and class label
    X = df.loc[:, df.columns != class_name].values
    y = df[class_name].values

    # 5. Step: preprare dataset dict
    dataset = {
        'name': datasetname,
        'df': df,
        'df_encoded': df,
        'columns': list(columns),
        'columns_for_decision_rules': list(columns_for_decision_rules),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': LabelEncoder(),
        'X': X,
        'y': y,
    }

    return dataset
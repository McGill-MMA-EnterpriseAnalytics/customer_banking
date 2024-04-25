import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate
import warnings 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
warnings.filterwarnings('ignore')  
def get_data(path):
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/aladelca/machine_learning_model/main/archivos_trabajo/bank-full.csv', sep=';')
    except:
        df = pd.read_csv(path, sep=';')
    return df

def map_target(df):
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    return df


def drop_columns(df):
    df = df.drop(['contact','poutcome', 'duration'], axis=1)
    return df

def preprocess_data(path):
    df = get_data(path)
    df = map_target(df)
    df = drop_columns(df)
    return df

if __name__=='__main__':
    df = preprocess_data('../Dataset/bank-full.csv')
    print(df)
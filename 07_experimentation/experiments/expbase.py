import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate
import warnings 
warnings.filterwarnings('ignore')   

df = pd.read_csv('https://raw.githubusercontent.com/aladelca/machine_learning_model/main/archivos_trabajo/bank-full.csv', sep=';')
df['y'] = df['y'].map({'yes': 1, 'no': 0})

### Delete columns leading to data leakage

df = df.drop(['contact','poutcome', 'duration'], axis=1)

mlflow.set_experiment("targeted_marketing")
with mlflow.start_run(run_name = 'catboost'):

    VARS = [
        'age', 
        'job', 
        'marital', 
        'education', 
        'balance', 
        'housing', 
        'loan', 
        'day', 
        'month', 
        'campaign', 
        ]
    TARGET = ['y']
    CAT_VARS = [
        'job',
        'marital',
        'education',
        'month',
        ]
    x = df[VARS]
    y = df[TARGET]

    x['housing'] = x['housing'].map({'yes': 1, 'no': 0})
    x['loan'] = x['loan'].map({'yes': 1, 'no': 0})

    ### Split the data into training and testing
    x_old, x_new, y_old, y_new = train_test_split(x, y, test_size=0.2, random_state=123)

    x_train, x_test, y_train, y_test = train_test_split(x_old, y_old, test_size=0.2, random_state=123)

    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=123)
    print('Training dataset:', x_fit.shape)
    print('Validation dataset:', x_val.shape)
    print('Test dataset:', x_test.shape)

    model = CatBoostClassifier(
        random_state=123, 
        cat_features = CAT_VARS, 
        verbose=0,
        eval_metric='AUC'
        )

    model.fit(x_fit, y_fit, early_stopping_rounds=10, eval_set=(x_val, y_val))
    mlflow.log_param('early_stopping_rounds', 10)
    mlflow.log_param('eval_metric', 'AUC')
    mlflow.log_metric('AUC_validation', roc_auc_score(y_val, model.predict_proba(x_val)[:,1]))
    mlflow.log_metric('AUC_test', roc_auc_score(y_test, model.predict_proba(x_test)[:,1]))
    mlflow.catboost.log_model(model, 'model_1')   
    x_fit.to_csv("training_data.csv")
    mlflow.log_artifact("training_data.csv")
    

mlflow.end_run()
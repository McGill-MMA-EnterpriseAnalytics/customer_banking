import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns, axis=1)


def load_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    return df


def preprocess_features(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in ['y']]

    # transformers for the numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # impute missing values with the mean (upon advanced imputing techniques we need to change this)
        ('scaler', MinMaxScaler()) 
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  #  simpleimputer constant strategy (upon advanced imputing techniques we need to change this)
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # one-hot encode categorical values
    ])

    # bundling preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor

# full preprocessing pipeline
def build_pipeline(df, drop_cols):
    pipeline = Pipeline(steps=[
        ('drop_columns', DropColumns(columns=drop_cols)),
        ('preprocessor', preprocess_features(df)),
        ('resampler', RandomOverSampler(random_state=0))  # handling class imbalance
    ])
    return pipeline

# split data
def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin
from fairml import audit_model
import matplotlib.pyplot as plt
from fairml import plot_dependencies
from imblearn.pipeline import Pipeline as ImbPipeline

# custom transformer to drop columns
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns, axis=1)

# load data
def load_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    return df

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns  # columns should be a list of column names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

def preprocess_features(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in ['y']]
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('to_df', DataFrameTransformer(numerical_cols))  # Convert back to DataFrame
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('to_df', DataFrameTransformer(categorical_cols))  # Convert back to DataFrame
    ])

    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ], remainder='passthrough')  # 'passthrough' to keep columns as DataFrame if untouched

    return preprocessor

""""
# Preprocess features
def preprocess_features(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in ['y']]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor
"""

# build the full preprocessing and modeling pipeline
def build_pipeline(df, drop_cols):
    pipeline = ImbPipeline(steps=[
        ('drop_columns', DropColumns(columns=drop_cols)),
        ('preprocessor', preprocess_features(df)),
        ('resampler', RandomOverSampler(random_state=0)),
        ('classifier', LogisticRegression(random_state=42))
    ])
    return pipeline


filepath = '/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/bank-full.csv'
df = load_data(filepath)
X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1), df['y'], test_size=0.2, random_state=42)

# Build and train the pipeline
drop_cols = [] # no columns to drop but for future if we want to drop columns
pipeline = build_pipeline(df, drop_cols)
pipeline.fit(X_train, y_train)



# Fairness analysis
# using the trained model for predictions and pass the whole test set for audit
model = pipeline.named_steps['classifier']
preprocessed_data = pipeline.named_steps['preprocessor'].transform(X_train)

# Audit the model
total, _ = audit_model(model.predict_proba, preprocessed_data)

# Plotting the results
fig = plot_dependencies(
    total.median(),
    title="FairML feature dependence",
    fig_size=(10, 8)
)
plt.show()

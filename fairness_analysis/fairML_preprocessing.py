import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from fairml import audit_model, plot_dependencies
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt


class DataFrameEnsurer(BaseEstimator, TransformerMixin):
    """Transforms an array back to a dataframe, ensuring proper column names after transformations."""
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.feature_names = None

    def fit(self, X, y=None):
        self.feature_names = self.preprocessor.get_feature_names_out()
        return self

    def transform(self, X):
        if self.feature_names is None:
            raise Exception("The transformer is not yet fitted with feature names.")
        return pd.DataFrame(X, columns=self.feature_names)

# Load and preprocess data
def preprocess_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    df = df.drop(['contact', 'poutcome', 'duration'], axis=1) # drop columns that are not informative

    # categorical and numerical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'y']

    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough')

    # Setup the full pipeline
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('to_df', DataFrameEnsurer(preprocessor)), 
        ('resampler', RandomOverSampler(random_state=0)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))  
    ])

    return df, pipeline


def perform_fairness_analysis(filepath):
    df, pipeline = preprocess_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1), df['y'], test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    model = pipeline.named_steps['classifier']
    preprocessed_data = pipeline.named_steps['to_df'].transform(pipeline.named_steps['preprocessor'].transform(X_train))

    # predict_proba
    predictions = model.predict_proba(preprocessed_data)[:, 1]  # probabilities for the positive class

    # Audit the model using the probabilities of the positive class
    total, _ = audit_model(lambda x: model.predict_proba(x)[:, 1], preprocessed_data) 


    fig = plot_dependencies(
        total.median(),
        title="FairML feature dependence",
        fig_size=(10, 8)
    )
    plt.savefig('fairness_analysis.png')  


if __name__ == '__main__':
    perform_fairness_analysis('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/bank-full.csv')

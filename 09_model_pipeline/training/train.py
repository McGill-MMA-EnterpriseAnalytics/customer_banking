
from sklearn.model_selection import train_test_split
from constants import *
from preprocess import preprocess_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from catboost import CatBoostClassifier

def get_variables(df):
    x = df[VARS]
    y = df[TARGET].values.ravel()
    return x,y

def get_onehot_encoder(x):
    x['housing'] = x['housing'].map({'yes': 1, 'no': 0})
    x['loan'] = x['loan'].map({'yes': 1, 'no': 0})
    return x

def train_model(path):
    print('Starting training model')
    df = preprocess_data(path)
    x, y = get_variables(df)
    x = get_onehot_encoder(x)
    print('Starting split ')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # Define the ordinal encoder transformation for categorical variables
    categorical_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder())
    ])

    # Column transformer for applying transformations to the specified columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, CAT_VARS)
        ],
        remainder='passthrough'
    )

    # Define the complete pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(random_state=123, verbose= False))
    ])

    # Fit the model
    model.fit(x_train, y_train)
    print('End fit')
    return model
    

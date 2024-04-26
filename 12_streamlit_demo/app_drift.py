import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

# KL Divergence
def calculate_kl_divergence(X_train, X_test, n_bins=10):
    kl_divergences = {}
    for feature in X_train.select_dtypes(include=[np.number]).columns:  # only numeric columns
        train_hist, bin_edges = np.histogram(X_train[feature], bins=n_bins, density=True)
        test_hist, _ = np.histogram(X_test[feature], bins=bin_edges, density=True)
        train_hist += 1e-10  # avoid division by zero
        test_hist += 1e-10
        train_hist /= train_hist.sum()
        test_hist /= test_hist.sum()
        kl_div = entropy(test_hist, train_hist)
        kl_divergences[feature] = kl_div
    return kl_divergences


# JS Divergence
def calculate_js_divergence(X_train, X_test, n_bins=10):
    js_divergences = {}
    for feature in X_train.columns:
        train_hist, bin_edges = np.histogram(X_train[feature], bins=n_bins, range=(0,1), density=True)
        test_hist, _ = np.histogram(X_test[feature], bins=bin_edges, density=True)
        js_div = jensenshannon(train_hist, test_hist, base=2)
        js_divergences[feature] = js_div
    return js_divergences

@st.cache  
def load_data():
    df = pd.read_csv('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/bank-full.csv', sep=';')
    numeric_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'] 
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # dropping columns that contain non-numeric values after coercion
    df.dropna(subset=numeric_cols, inplace=True)
    return df


df = load_data()
X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1), df['y'], test_size=0.2, random_state=42)

print(X_train.dtypes)
kl_divergences = calculate_kl_divergence(X_train, X_test)

# Calculate divergences
kl_divergences = calculate_kl_divergence(X_train, X_test)
js_divergences = calculate_js_divergence(X_train, X_test)

# ui
st.title('Data Drift Analysis Dashboard')

st.write("## Data Overview")
st.write(df.describe())

st.write("## Target Variable Distribution")
fig, ax = plt.subplots()
sns.histplot(df['y'], ax=ax)
st.pyplot(fig)

st.write("## KL Divergence")
kl_df = pd.DataFrame(list(kl_divergences.items()), columns=['Feature', 'KL Divergence'])
st.write(kl_df)

st.write("## Jensen-Shannon Divergence")
js_df = pd.DataFrame(list(js_divergences.items()), columns=['Feature', 'JS Divergence'])
st.write(js_df)

st.write("## Interactive Data Table")
st.dataframe(df)

st.write("## Feature Selection")
selected_feature = st.selectbox('Select feature for detailed analysis:', df.columns)
st.write(f"You selected: {selected_feature}")

# histograms for selected feature
fig, ax = plt.subplots()
sns.histplot(X_train[selected_feature], ax=ax, color='blue', label='Train')
sns.histplot(X_test[selected_feature], ax=ax, color='red', label='Test')
ax.legend()
st.pyplot(fig)

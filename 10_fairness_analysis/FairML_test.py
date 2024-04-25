import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairml import audit_model

# Create some sample data
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [4, 3, 2, 1],
    'target': [0, 1, 0, 1]
})

# Fit a model
clf = LogisticRegression()
clf.fit(data[['feature1', 'feature2']], data['target'])

# Audit the model
total, _ = audit_model(clf.predict, data[['feature1', 'feature2']])
print(total)

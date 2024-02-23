# Predicting Subscription Outcomes in Direct Marketing Campaigns

## Team members
Adrian Alarcon Delgado, Kritika Nayyar, Mohamed Elenany, Sheida Majidi, Vincent El-Ghoubaira

## Objective
Our primary objective is to predict the probability of converting a client as a result of marketing campaign initiatives. The classification objective revolves around predicting whether a client will subscribe (yes/no) to a term deposit (variable y). Additionally, alongside predictive modeling, we have delved into uplift modeling (prescriptive modeling) to assess the potential impact of company actions on customer outcomes.

## About the data
The dataset is derived from direct marketing campaigns conducted by a Portuguese banking institution. These campaigns involved phone calls, often requiring multiple contacts with the same client to assess whether they would subscribe to the bank’s term deposit product.

## Approach
### Classification task
#### Pre-processing
1. Feature Selection:
- We removed the “poutcome” feature due to its lack of predictive power.
- The “contact” feature was redundant and was also removed.
- High multicollinearity was observed between the client’s age and the day of contact. Consequently, we dropped the “day” feature. Our hypothesis is that maybe certain calls were made for special occasions (e.g., birthdays), but these were not strongly correlated with conversion outcomes.
2. Data Transformation:
- We performed categorical encoding.
- An isolation forest was used to identify potential outliers.
- Employed RandomForestClassifier with a focus on using full data.
- Job type did not significantly impact conversion outcomes.
- Addressing class imbalance, we employed an over-sampling technique.
- Min-Max Scaler was applied to normalize features.

#### Model
Logistic, Random, Gradient, LightGBM, XGBoost. We go with Scaled one. We focus on most positives - recall. Hence, Gradient Boosting. If client was contacted before, XX. 
Organic in Mar and Sept (They approached). Wrong time of campaign in May - Aug. Apr is tax season. Marketing campaign after tax.
threshold of 0.5. PR curve (most efficient threshold?)
Useful to perform cross-validation.
Gradient Boosting over Logistic.

### Uplift Modelling
Target is conversion.
Treatment is poutcome
LinearDML
ForestMDL (confidence > kinear)
Causal only on success or failure.

Train model for success + train model for failure.
ROC curve. Predicting the prob of success.




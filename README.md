# Predicting subscription outcome of direct marketing campaigns

## Team members
Adrian Alarcon Delgado, Kritika Nayyar, Mohamed Elenany, Sheida Majidi, Vincent El-Ghoubaira

## Objective
Our goal is to predict probability of converting a client as a result of marketing campaign initiatives. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y). 

In addition to predictive modelling, we have also worked on uplift modelling (prescriptive modelling) to estimate the impact of an company action on customer outcome. 

## About the data
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

## Approach
### Classification task
#### Pre-processing
Removed poutcome since no predictive power. Contact removed - redundant. Age and day of customer had a very high VIF. Removed day. Sometimes bank call for birthday etc. 
categorical encoding. Isolation forest. Job type did not have a correction with conversion outcome. Used RandomForestClassifier > full data should be used. Class imbalance. Over-sampler > under-sample. Min-Max Scaler fitting to training. 

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




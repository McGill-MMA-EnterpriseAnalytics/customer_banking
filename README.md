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
- We evaluated several classification models:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. LightGBM
5. XGBoost
- We considered whether a client had been contacted previously.

Model Selection:
- Considering our focus on maximizing recall (identifying true positives), we selected the scaled Gradient Boosting model.
- Preferred Gradient Boosting over Logistic Regression for its superior performance.

Timing Considerations:
- Organic marketing efforts were most effective in March and September.
- Campaigns conducted between May and August were less successful.
- April, being tax season, influenced customer behavior.
- Post-tax marketing campaigns were strategically timed.

### Uplift Modelling
#### Approach 1:
Our target variable is conversion.
The treatment variable is “poutcome.”
Employed LinearDML and ForestMDL, prioritizing confidence in predictions.
Focused solely on causal effects concerning success or failure outcomes.

#### Approach 2:
Trained separate models for success and failure scenarios.
Assessed model performance using ROC curve to predict the probability of success.




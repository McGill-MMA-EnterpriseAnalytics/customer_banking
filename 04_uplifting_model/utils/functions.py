from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def get_datasets(data,vars, target):
    x = data[vars]
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    x_val = x_train[vars].sample(int(round(x_train.shape[0]*0.1,0)), random_state=123)
    y_val = y_train.loc[x_val.index]
    x_train = x_train.drop(x_val.index)
    y_train = y_train.drop(x_val.index)
    return x_train, x_val, x_test, y_train, y_val, y_test


def get_predictions(model, threshold, x_test):
    y_proba = model.predict_proba(x_test)
    y_pred = np.where(y_proba[:,1] > threshold, 1, 0)
    return y_pred, y_proba

def training_process(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=5, verbose=False)
    
    y_val, y_proba_val = get_predictions(model, 0.5, x_val)
    return model, y_val, y_proba_val


def get_metrics(y_test, y_pred, y_proba):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:,1])
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    return metrics

def predict_process(model, threshold, x_test, y_test):
    y_pred, y_proba = get_predictions(model, threshold, x_test)
    metrics = get_metrics(y_test, y_pred, y_proba)
    return y_pred, y_proba, metrics


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, threshold = roc_curve(y_test, y_proba[:,1])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return fpr, tpr, threshold

def plot_precision_recall(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    
    auc_precision_recall = auc(recall, precision)

    
    plt.figure()
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precisión-Recall')
    plt.legend(loc='best')
    plt.show()
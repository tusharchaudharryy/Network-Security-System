<<<<<<< HEAD
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param, scoring='roc_auc'):
    """
    Evaluate classification models using ROC-AUC by default.
    Returns a report dict: {model_name: test_roc_auc}
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, scoring=scoring, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict probabilities for ROC-AUC
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]

            train_model_score = roc_auc_score(y_train, y_train_pred_proba)
            test_model_score = roc_auc_score(y_test, y_test_pred_proba)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
=======
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param, scoring='roc_auc'):
    """
    Evaluate classification models using ROC-AUC by default.
    Returns a report dict: {model_name: test_roc_auc}
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, scoring=scoring, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict probabilities for ROC-AUC
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]

            train_model_score = roc_auc_score(y_train, y_train_pred_proba)
            test_model_score = roc_auc_score(y_test, y_test_pred_proba)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
>>>>>>> 568bd63 (New Commits)

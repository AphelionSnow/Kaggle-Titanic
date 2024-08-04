from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score, roc_curve, auc, f1_score
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # optimized RF model
from scipy.stats import randint
from itertools import combinations # combinations of vars

class InputSampler():
    def __init__(self, cat, num, data) -> None:
        # This class will be used to go through the steps of testing performance 
        # of different regression models using subsets of provided variables.
        self.num = num
        self.cat = cat
        self.var_combinations = self._var_combinations()
        self.bounded_combinations = None
        
        self.X = data[cat+num]
        self.y = data['Survived'] # only relevant for this specific project
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.ct = Pipeline(steps=[ # categorical transformer
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.nt = Pipeline(steps=[ # numerical transformer
            ('scaler', StandardScaler())
        ])


    def sampleRFC(self):
        # TODO 2
        # RandomForest model. Returns list of models with highest cv scores.
        pass
    
    def sampleLogR(self):
        # TODO 1
        # LogisticRegression model. Returns cv scores for optimized Logistic Regression model.
        pass
    
    def sampleFull(self):
        # TODO 3
        # Sample on all models and combine resulting arrays.
        # returns results sorted by mean cv score.
        pass
    
    def _logr_find_threshold(self, model):
        # Finds optimal threshold for Logistic Regression model.
        y_prob = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds_roc = roc_curve(self.y_test, y_prob)
        
        distances = np.sqrt(fpr**2 + (1 - tpr)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds_roc[optimal_idx]
        return optimal_threshold
    
    def _var_combinations(self):
        # create a list of all combinations of provided variables, split by data type
        # output the combinations to self.combinations on initialization of object
        var_combinations_tuples = []
        for r in range(len(self.num)+len(self.cat)):
            var_combinations_tuples.append(combinations((self.num+self.cat), r))
        var_combinations = []
        for tuple in var_combinations_tuples:
            num_vars = []
            cat_vars = []
            vars = list(tuple)
            for var in vars:
                if var in self.num:
                    num_vars.append(var)
                elif var in self.cat:
                    num_vars.append(var)
            var_combinations.append([num_vars, cat_vars])
        return var_combinations     
    
    def setRequired(self, req):
        # set variables that are required to be used in every sampling combination. Must be a list input.
        # initiates self.bounded_combinations as a subset of self.combinations that include all required variables
        if type(req) != list:
            print('Argument type must be list')
            return
        
        bounded = []
        for combination in self.var_combinations:
            # see if all required cars are in the combination
            exists = True
            for var in req:
                if var not in combination[0] and var not in combination[1]:
                    exists = False
                    break
            if exists:
                bounded.append(combination)
        self.bounded_combinations = bounded
    
    def _preprocess(self, num, cat):
        # helper function for preprocessor handling
        preprocessor = None
        if not num:
            preprocessor = ColumnTransformer(
            transformers=[
                ('cat', self.ct, cat)
                ]
            )
        elif not cat:
            preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.nt, num)
                ]
            )
        else:
            preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.nt, num)
                ('cat', self.ct, cat)
                ]
            )
        
        return preprocessor


    
    

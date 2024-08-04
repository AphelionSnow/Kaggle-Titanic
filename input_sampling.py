from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score, roc_curve, auc
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # optimized RF model
from scipy.stats import randint
import itertools # combinations of vars

class InputSampler():
    def __init__(self, cat, num, X, y) -> None:
        """
        This class will be used to go through the steps of testing performance of different regression models 
        using subsets of provided variables. GridSearchCV/RandomizedSearchCV for optimal arguments (this will take time)
        """
        self.num = num
        self.cat = cat
        self.combinations = []
        self.bounded_combinations = None
        
        self.X = X
        self.y = y
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.model = None
        self.ct = Pipeline(steps=[ # categorical transformer
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.nt = Pipeline(steps=[ # numerical transformer
            ('scaler', StandardScaler())
        ])
        
        
    def sampleRFC(self):
        # RandomForest model
        # preprocessor = ColumnTransformer(
        # transformers=[
        #     ('num', self.nt, numerical),
        #     ('cat', self.ct, categorical)
        #     ]
        # )
        pass
    
    def sampleLogR(self):
        # LogisticRegression model
        pass
    
    cat_trans = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_trans = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    def sampleFull(self):
        # Sample on all models and combine resulting arrays
        # returns results sorted by mean cv score
        pass
    
    def _var_combinations(self):
        # create a list of all combinations of provided variables
        # output the combinations to self.combinations on initialization of object
        pass
    
    def setRequired(self, req):
        # set variables that are required to be used in every sampling combination. Must be a list input.
        # initiates self.bounded_combinations as a subset of self.combinations that include all required variables
        if type(req) != list:
            print('Argument type must be list')
            return
        
        bounded = []
        for combination in self.combinations:
            # see if all required cars are in the combination
            exists = True
            for var in req:
                if var not in combination[0] and var not in combination[1]:
                    exists = False
                    break
            if exists:
                bounded.append(combination)
        self.bounded_combinations = bounded
                
    
    def _split_train_test(self):
        # split the X and y 
        pass
    
    def _preprocess(self):
        # helper function for preprocessor to make code more readable
        pass
    
    def _combination_decoder(self):
        # splits combinations back into numerical and categorical classifications
        # deprecating this function by resplitting at time of combination
        pass


    
    

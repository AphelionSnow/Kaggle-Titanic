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
from sklearn.base import BaseEstimator, TransformerMixin

class InputSampler():
    def __init__(self, num, cat, data) -> None:
        # This class will be used to go through the steps of testing performance 
        # of different regression models using subsets of provided variables.
        self.num = num
        self.cat = cat
        self.data = data
        self.var_combinations = self._var_combinations()
        self.bounded_combinations = None
        
        # self.X = self.data[cat+num]
        # self.y = data['Survived']
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[num+cat], self.data['Survived'], test_size=0.2, random_state=42)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # self.ct = Pipeline(steps=[ # categorical transformer
        # ('onehot', OneHotEncoder(handle_unknown='ignore'))
        # ])
        # self.nt = Pipeline(steps=[ # numerical transformer
        #     ('scaler', StandardScaler())
        # ])


    def sampleRFC(self):
        # TODO 2
        # RandomForest model. Returns list of models with highest cv scores.
        pass
    
    def sampleLogR(self):
        # TODO 1
        # LogisticRegression model sampling. 
        # Returns list of cv scores for Logistic Regression fits after checking every combination of variables. Sorted descending.
        # Future idea: test for optimal train/test split ratio
        # Maybe use cv scores later if logistic regression competes with other models
        # cv_scores = []
        
        # if self.bounded_combinations:
        #     for combination in self.bounded_combinations:
                
        #         # Debugging: Check the shapes and content of X_train before fitting
        #         # print("Shape of X_train:", self.X_train.shape)
        #         # print("First few rows of X_train:\n", self.X_train.head())
        #         # print("Categorical features: ", combination[0])
        #         # print("Numerical features: ", combination[1])
                
        #         # preprocessor = self._preprocess(combination[0], combination[1])
        #         # model = Pipeline(steps=[('preprocessor', preprocessor),
        #         #                         ('classifier', LogisticRegression())
        #         # ])
        #         model = LogisticRegression()
        #         model.fit(self.X_train, self.y_train)
        #         y_prob = model.predict_proba(self.X_test)[:, 1]
        #         y_pred = (y_prob >= self._logr_find_threshold(y_prob)).astype(int)
        #         accuracy = accuracy_score(self.y_test, y_pred)
        #         cv_scores.append([accuracy, combination])
        # else:
        #     for combination in self.var_combinations:
                
        #         # Debugging: Check the shapes and content of X_train before fitting
        #         # print("Categorical features: ", combination[1])
        #         # print("Numerical features: ", combination[0], '\n')
                
        #         # preprocessor = self._preprocess(combination[1], combination[0])
        #         # model = Pipeline(steps=[('preprocessor', preprocessor),
        #         #                         ('classifier', LogisticRegression())
        #         # ])
                
        #         # print("Combination:", combination)
        #         # print(X_train)
        #         model = LogisticRegression()
        #         model.fit(self.X_train, self.y_train)
        #         y_prob = model.predict_proba(self.X_test)[:, 1]
        #         y_pred = (y_prob >= self._logr_find_threshold(y_prob)).astype(int)
        #         accuracy = accuracy_score(self.y_test, y_pred)
        #         cv_scores.append([accuracy, combination])
                
        # return sorted(cv_scores, key=lambda x:x[0], reverse=True)
        
        results = []
        for combo in self.var_combinations:
            num_vars = combo[0]
            cat_vars = combo[1]
            pipeline = self._create_pipeline(num_vars, cat_vars)
            
            # Fit the pipeline on the training data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[num_vars+cat_vars], self.data['Survived'], test_size=0.2, random_state=42)
            pipeline.fit(self.X_train[num_vars+cat_vars], self.y_train)
            
            # Predict probabilities and find optimal threshold
            probas_ = pipeline.predict_proba(self.X_test[num_vars+cat_vars])[:, 1]
            optimal_threshold = self._logr_find_threshold(probas_)
            
            # Calculate the score with the optimal threshold
            predicted = (probas_ >= optimal_threshold).astype(int)
            score = np.mean(predicted == self.y_test)
            results.append([score, pipeline, combo, optimal_threshold])
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
        
        return sorted(results, key=lambda x:x[0], reverse=True)
    
    def sampleFull(self):
        # TODO 3
        # Sample on all models and combine resulting arrays.
        # returns results sorted by mean cv score.
        pass
    
    def _logr_find_threshold(self, y_prob):
        # Finds optimal threshold for Logistic Regression model.
        fpr, tpr, thresholds_roc = roc_curve(self.y_test, y_prob)
        
        # NOTE: try this to find optimal threshold, otherwise try to use j-score
        distances = np.sqrt(fpr**2 + (1 - tpr)**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds_roc[optimal_idx]
        
        return optimal_threshold
    
    def _var_combinations(self):
        # create a list of all combinations of provided variables, split by data type
        # output the combinations to self.combinations on initialization of object
        var_combinations_tuples = []
        for r in range(len(self.num)+len(self.cat)-1):
            var_combinations_tuples.extend(combinations((self.num+self.cat), r+1))
        var_combinations = []
        # print(var_combinations_tuples)
        for tuple in var_combinations_tuples:
            num_vars = []
            cat_vars = []
            vars = list(tuple)
            # print('vars', vars)
            for var in vars:
                # print('var', var)
                if var in self.num:
                    num_vars.append(var)
                elif var in self.cat:
                    cat_vars.append(var)
            # print(num_vars, cat_vars)
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
    
    # def _preprocess(self, num, cat):
    #     # helper function for preprocessor handling
    #     preprocessor = None
    #     if not num:
    #         preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('cat', self.ct, cat)
    #             ]
    #         )
    #     elif not cat:
    #         preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('num', self.nt, num)
    #             ]
    #         )
    #     else:
    #         preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('num', StandardScaler(), num)
    #             ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
    #             ]
    #         )
    #     return preprocessor
    
    def _create_pipeline(self, num_vars, cat_vars):
        feature_selector = ColumnSelector(columns=num_vars + cat_vars)
        
        # Create a preprocessing pipeline
        transformers = []
        if num_vars:
            transformers.append(('scaler', StandardScaler(), num_vars))
        if cat_vars:
            transformers.append(('onehot', OneHotEncoder(), cat_vars))
    
        preprocessor = ColumnTransformer(transformers=transformers)
        pipeline = Pipeline(steps=[
            # ('selector', feature_selector),
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=500))
        ])
        return pipeline
    
    def _encode_data(self, data):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat)
            ]
        )
        return preprocessor.fit_transform(data)


    
# GPT code
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
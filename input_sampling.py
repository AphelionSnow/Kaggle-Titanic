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
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def sampleRFC(self):
        # TODO 1
        # RandomForest model. Returns list of models with highest cv scores.
        # Will implement bounded_combinations variation later
        param_dist = { # 'classifier__' prefix allows use in pipeline
            'classifier__n_estimators': [int(x) for x in range(100, 1200, 100)],
            'classifier__max_features': [None, 'sqrt', 'log2'],
            'classifier__max_depth': [int(x) for x in range(10, 110, 10)] + [None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__bootstrap': [True, False],
            'classifier__criterion': ['gini', 'entropy']
        }
        
        highest = 0
        iteration = 0
        
        results = []
        for combo in self.var_combinations:
            num_vars = combo[0]
            cat_vars = combo[1]
            pipeline = self._create_pipeline(num_vars, cat_vars, 'Forest')
            pipeline = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist,
                                              n_iter=250, cv=5, verbose=2, random_state=42, n_jobs=-1)
            
            # Fit the pipeline on the training data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[num_vars+cat_vars], self.data['Survived'], test_size=0.2, random_state=42)
            pipeline.fit(self.X_train[num_vars+cat_vars], self.y_train)

            # Retrieve the best parameters and model
            best_params = pipeline.best_params_
            best_rf = pipeline.best_estimator_
            
            # Return score with the optimal parameters
            score = best_rf.score(self.X_test, self.y_test)
            results.append([score, pipeline, combo, [pipeline.best_params_, pipeline.best_score_, pipeline.best_estimator_]])
            print([score, pipeline, combo, [pipeline.best_params_, pipeline.best_score_, pipeline.best_estimator_]])
            print('Highest =', highest)
            if score > highest:
                highest = score
            iteration += 1
            print('Iteration =', iteration)
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            
        return results
    
    def sampleLogR(self):
        # LogisticRegression model sampling. 
        # Returns list of cv scores for Logistic Regression fits after checking every combination of variables. Sorted descending.
        # Future idea: test for optimal train/test split ratio
        # Maybe use cv scores later if logistic regression competes with other models

        results = []
        for combo in self.var_combinations: # will implement bounded_combinations later
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
        # TODO 2
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
        for r in range(2):
            var_combinations_tuples.extend(combinations((self.num+self.cat), r+6))
            
        var_combinations = []
        for tuple in var_combinations_tuples:
            num_vars = []
            cat_vars = []
            vars = list(tuple)
            for var in vars:
                if var in self.num:
                    num_vars.append(var)
                elif var in self.cat:
                    cat_vars.append(var)
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
    
    def _create_pipeline(self, num_vars, cat_vars, type='Log'):
        
        # Create a preprocessing pipeline
        transformers = []
        if num_vars:
            transformers.append(('scaler', StandardScaler(), num_vars))
        if cat_vars:
            transformers.append(('onehot', OneHotEncoder(), cat_vars))
    
        preprocessor = ColumnTransformer(transformers=transformers)
        if type == 'Log':
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=500))
            ])
        elif type == 'Forest':
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
        return pipeline
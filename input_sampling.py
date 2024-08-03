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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

class InputSampler():
    def __init__(self, cat, num) -> None:
        """
        This class will be used to go through the steps of testing performance of different regression models 
        using subsets of provided variables. GridSearchCV/RandomizedSearchCV for optimal arguments (this will take time)
        """
        self.num = num
        self.cat = cat
        self.model = None
    
    def sampleRFC(self):
        # RandomForest model
        pass
    
    def sampleLogR(self):
        # LogisticRegression model
        pass
    

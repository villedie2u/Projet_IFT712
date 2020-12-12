# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:30:58 2020

@author: maxim
"""
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import time
from humanfriendly import format_timespan
import warnings
warnings.filterwarnings("ignore")

class RegLog:
    def __init__(self):
        self.predictions = []
        self.classification_report = 0
        self.accuracy_score = 0

    def reg_log(self, data_train, data_target):
        paramC = int(input("Parameter C tuning: 0 -> No, 1 -> Yes : "))
        if (paramC == 0):
            print("Training model...")
            start = time.time()
            x_train, x_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)
            
            model = LogisticRegression()
            model.fit(x_train, y_train)
            
            end = time.time()
            time_spent = end - start
            
            print("Time taken for training and testing : ",format_timespan(time_spent)," seconds")
    
            self.predictions = model.predict(x_test)
            self.classification_report = classification_report(y_test, self.predictions)
            self.accuracy_score = accuracy_score(y_test, self.predictions)
    
            self.print_results()
            
        else:
            
        
            print("Please, choose a number of possible C parameters for the model to choose from during the tuning phase.")
            print("Depending on the number you choose, this phase can be relatively long (between a few seconds to more than three if you pick 60 C for example).")
            print("We advise you to choose 50 for C. Expected waiting time : about 3 minutes.")
            nb_Cs = int(input("Number of C : "))
            Cs = np.logspace(start = 0.001, stop = 4, num = nb_Cs)
            print("CS : ",Cs)
            
            param_grid = {'C': Cs}
            print("C parameter tuning. Please, wait. It could take a few minutes.")
            start = time.time()
            
            logisticReg=LogisticRegression()
            grid_search = GridSearchCV(logisticReg, param_grid, cv=10)
            grid_search.fit(data_train, data_target)
            best = grid_search.best_params_
            
            end = time.time()
            time_spent = end - start
            
            print("Time taken : ",format_timespan(time_spent)," seconds")
            
            print("BEST PARAMS: ",best)
            
            print(float(best['C']))
            
            print("Training model with the best parameters found...")
            x_train, x_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)
            
            model = LogisticRegression(C=float(best['C']))
            model.fit(x_train, y_train)
    
            self.predictions = model.predict(x_test)
            self.classification_report = classification_report(y_test, self.predictions)
            self.accuracy_score = accuracy_score(y_test, self.predictions)
    
            self.print_results()

    def print_results(self):
        print("predictions:", self.predictions, "\n")
        print("classification_report:", self.classification_report, "\n")
        print("accuracy_score:", self.accuracy_score, "\n")

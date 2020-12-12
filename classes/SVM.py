# -*- coding: utf-8 -*-
"""
class using the SVM model to predict targets
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import time
from humanfriendly import format_timespan
import warnings
warnings.filterwarnings("ignore")


class SupportVectorMachine:
    
    def __init__(self):
        self.predictions = []
        self.classification_report = 0
        self.accuracy_score = 0

    def Svm(self, data_train, data_target):
        print("Number of data points:", data_train.shape[0])
        print("Number of features:", data_train.shape[1])
        
        print("Choose a kernel: linear, rbf or poly")
        opt = input("Kernel : ")
        
        good_option = 0
        while good_option == 0:
            if opt != 'linear' and opt != 'rbf' and opt != 'poly':
                print("Not recognized kernel, please try again.")
                opt = input("Kernel : ")
            else:
                good_option = 1

        Cchoice = int(input("Choose an option. 0 : No C parameter tuning. 1 : C parameter tuning. : "))
        if Cchoice == 0:
            print("Training model with a ", opt, " kernel")
            
            X_train, X_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)
            
            if opt == 'poly':
                deg = int(input("Please, choose a degree for the polynomial kernel (such as 3): "))
                smodel = SVC(degree=deg)
            else:
                smodel = SVC()
        
            smodel.fit(X_train, y_train)
            self.predictions = smodel.predict(X_test)
            self.classification_report = classification_report(y_test, self.predictions)
            self.accuracy_score = accuracy_score(y_test, self.predictions)
    
            self.print_results()
        else:
            print("Please, choose a number of possible C parameters for the model "
                  "to choose from during the tuning phase.")
            print("Depending on the number you choose, this phase can be relatively long "
                  "(between a few seconds to more than five if you pick 40 C for example).")
            print("We advise you to choose 30 for C. Expected waiting time : about 4 minutes.")

            nb_Cs = int(input("Number of C : "))
            Cs = np.logspace(start=0.001, stop=3, num=nb_Cs)
            print("CS :", Cs)
            
            param_grid = {'C': Cs}
            
            print("C parameter tuning. Please, wait. It could take a few minutes.")
            start = time.time()
            
            grid_search = GridSearchCV(SVC(kernel=opt), param_grid, cv=10)
            grid_search.fit(data_train, data_target)
            best = grid_search.best_params_
            
            end = time.time()
            time_spent = end - start
            
            print("Time taken :", format_timespan(time_spent), "seconds")
            
            print("BEST PARAMS:", best)
            
            print(float(best['C']))
            
            print("Training model with the best parameters found and a", opt, "kernel")
            
            X_train, X_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)
            
            if opt == 'poly':
                deg = int(input("Please, choose a degree for the polynomial kernel (such as 3): "))
                smodel = SVC(C=float(best['C']), kernel=opt, degree=deg)
            else:
                smodel = SVC(C=float(best['C']), kernel=opt)
        
            smodel.fit(X_train, y_train)
            self.predictions = smodel.predict(X_test)
            self.classification_report = classification_report(y_test, self.predictions)
            self.accuracy_score = accuracy_score(y_test, self.predictions)
    
            self.print_results()
        
    def print_results(self):
        print("predictions:", self.predictions, "\n")
        print("classification_report:", self.classification_report, "\n")
        print("accuracy_score:", self.accuracy_score, "\n")
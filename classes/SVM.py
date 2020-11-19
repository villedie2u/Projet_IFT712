# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:15:26 2020

@author: maxim
"""
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import time
from humanfriendly import format_timespan
import warnings
warnings.filterwarnings("ignore")

class SupportVectorMachine:
    
    def __init__(self):
        self.predictions = []
        self.classification_report = 0
        self.accuracy_score = 0
        
    
    def Svm(self,data_train,data_target):
        print ("Number of data points ::", data_train.shape[0])
        print("Number of features ::", data_train.shape[1])
        
        #Methode internet 1 => meilleur C trouvé : 0.74 et quelques avec un score de 94
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_train)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, data_target, test_size=0.2, random_state=42)
        
        clf = LinearSVC(penalty='l2', loss='squared_hinge',
                        dual=True, tol=0.0001, C=100, multi_class='ovr',
                        fit_intercept=True, intercept_scaling=1, class_weight=None,verbose=0
                        , random_state=0, max_iter=1000)
        clf.fit(X_train,y_train)
        
        print('Accuracy of linear SVC on training set: {:.2f}'.format(clf.score(X_train, y_train)))
        
        print('Accuracy of linear SVC on test set: {:.2f}'.format(clf.score(X_test, y_test)))
                
        c = np.logspace(start = -15, stop = 1000, base = 1.02)
        param_grid = {'C': c}
        
        
        grid = GridSearchCV(clf, param_grid =param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)
          
        print("The best parameters are %s with a score of %0.0f" % (grid.best_params_, grid.best_score_ * 100 ))
        print( "Best estimator accuracy on test set {:.2f} ".format(grid.best_estimator_.score(X_test, y_test) * 100 ) )   
        """
        
        #Internet methode 2 (sklearn SVM tuto)
        """
        X_train, X_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)
        
        
        option = int(input("1:Linear, 2:RBF, 3:Polynomial. => "))
        option = int(option)
        if (option == 1):
            smodel = SVC(kernel="linear")
            
        elif (option == 2):
            smodel = SVC(kernel="rbf")
            #best_parm = self.SVC_param_tuning()
            #print("BEST PARAM : ",best_parm)
        elif (option == 3):
            degre = int(input("Degree : "))
            smodel = SVC(kernel="poly",degree=degre)
        else:
            print("option non reconnue")
        
        
        smodel.fit(X_train, y_train)
        """
        
        #Tuning parameters C et gamma
        
        
        #print("Accuracy:",accuracy_score(y_test, y_pred))
        """
        self.predictions = smodel.predict(X_test)
        self.classification_report = classification_report(y_test, self.predictions)
        self.accuracy_score = accuracy_score(y_test, self.predictions)

        self.print_results() 
        """
        
        """
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        """
        
        print("Choose a kernel: linear, rbf or poly")
        opt = input("Kernel : ")
        
        good_option = 0
        while (good_option == 0):
            if (opt != 'linear' and opt != 'rbf' and opt != 'poly'):
                print("Not recognized kernel, please try again.")
                opt = input("Kernel : ")
            else:
                good_option = 1
        
        #print("Please, choose a number of possible C and Gamma parameters for the model to choose from during the tuning phase.")
        #print("Depending on the number you choose, this phase can be relatively long (between one or two minutes to more than ten if you pick 10 C and 10 Gammas for example).")
        #print("We advise you to choose 5 for C and 5 for Gamma. Expected waiting time : about 2 or 3 minutes.")
        
        print("Please, choose a number of possible C parameters for the model to choose from during the tuning phase.")
        print("Depending on the number you choose, this phase can be relatively long (between a few seconds to more than five if you pick 40 C for example).")
        print("We advise you to choose 30 for C. Expected waiting time : about 4 minutes.")
        
        
        nb_Cs = int(input("Number of C : "))
        #nb_Gammas = int(input("Number of Gammas : "))
        Cs = np.logspace(start = 0.001, stop = 3, num = nb_Cs)
        #gammas = np.logspace(start = 0.001, stop = 3, num = nb_Gammas)
        #print("CS : ",Cs," GAMMA : ",gammas)
        print("CS : ",Cs)
        
        #param_grid = {'C': Cs, 'gamma' : gammas}
        param_grid = {'C': Cs}
        
        #print("C and Gamma parameters tuning. Please, wait. It could take a few minutes.")
        print("C parameter tuning. Please, wait. It could take a few minutes.")
        start = time.time()
        
        grid_search = GridSearchCV(SVC(kernel=opt), param_grid, cv=10)
        grid_search.fit(data_train, data_target)
        best = grid_search.best_params_
        
        end = time.time()
        time_spent = end - start
        
        print("Time taken : ",format_timespan(time_spent)," seconds")
        
        #print("BEST PARAMS: ",best," TYPE : ",type(best))
        print("BEST PARAMS: ",best)
        
        #print(float(best['C'])," TYPE : ",type(float(best['Gamma'])))
        print(float(best['C']))
        
        print("Training model with the best parameters found and a ",opt," kernel")
        
        X_train, X_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)
        
        if (opt == 'poly'):
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
        
    
    def svc_param_tuning(X,y,ker,nfolds):
        Cs = np.logspace(start = -10, stop = 100)
        gammas = np.logspace(start = 0.001, stop = 1)
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(SVC(kernel=ker), param_grid, cv=10)
        grid_search.fit(X, y)
        grid_search.best_params_
        
        return grid_search.best_params_
    
    def SVC_param_tuning():
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=10)
        grid_search.fit(data_train, data_target)
        grid_search.best_params_
        return grid_search.best_params_
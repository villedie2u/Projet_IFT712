# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:19:32 2020

@author: maxim
"""
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class KNNeighbours:
    
    def __init__(self):
        self.predictions = []
        self.classification_report = 0
        self.accuracy_score = 0
    
    def Knn(self,data_train,data_target):
        
        # Determination of the best K parameter
        K_list = []
        for i in range(1,51):
            K_list.append(i)
            
        scores_list = []
        
        # 10-fold cross validation
        for k in K_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, data_train, data_target, cv=10, scoring='accuracy')
            scores_list.append(scores.mean())
            #print("Pour k =",k," on a erreur moyenne = ",1-scores.mean())
            
        # turning this into a misclassification errors list
        Errors = []
        for i in scores_list:
            Errors.append(1-i)
        
        plt.figure()
        plt.figure(figsize=(15,10))
        plt.title('Optimal number of neighbors', fontsize=20, fontweight='bold')
        plt.xlabel('K Parameter', fontsize=15)
        plt.ylabel('Misclassification Error', fontsize=15)
        plt.plot(K_list, Errors)
        
        plt.show()
        
        # finding best k
        best_k = K_list[Errors.index(min(Errors))]
        
        # Training and evaluating the model's accuracy with the best k we found
        x_train,x_test,y_train,y_test=train_test_split(data_train,data_target,test_size=0.2)
        neigh=KNeighborsClassifier(n_neighbors=best_k)
        neigh.fit(x_train,y_train)
        
        self.predictions = neigh.predict(x_test)
        self.classification_report = classification_report(y_test, self.predictions)
        self.accuracy_score = accuracy_score(y_test, self.predictions)

        self.print_results()  
        
        print("The optimal number of neighbors is %d." % best_k)
        
    def print_results(self):
        print("predictions:", self.predictions, "\n")
        print("classification_report:", self.classification_report, "\n")
        print("accuracy_score:", self.accuracy_score, "\n")

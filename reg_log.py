# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:30:58 2020

@author: maxim
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def reg_log(X_train,X_target):
    """ Dans le main, X_train est df_train et X_target est df_target """
    
    x_train, x_test, y_train, y_test = train_test_split(X_train, X_target, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    print(predictions)
    print()
    
    print( classification_report(y_test, predictions) )
    
    print( accuracy_score(y_test, predictions))
    
    
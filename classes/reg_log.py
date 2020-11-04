# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:30:58 2020

@author: maxim
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RegLog:
    def __init__(self):
        self.predictions = []
        self.classification_report = 0
        self.accuracy_score = 0

    def reg_log(self, data_train, data_target):
        """ Dans le main, X_train est df_train et X_target est df_target """

        x_train, x_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(x_train, y_train)

        self.predictions = model.predict(x_test)
        self.classification_report = classification_report(y_test, self.predictions)
        self.accuracy_score = accuracy_score(y_test, self.predictions)

        self.print_results()

    def print_results(self):
        print("predictions:", self.predictions, "\n")
        print("classification_report:", self.classification_report, "\n")
        print("accuracy_score:", self.accuracy_score, "\n")

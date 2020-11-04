""" class trying to learn how to identify the class of an item using the perceptron method"""

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split


class Percep:

    def __init__(self, lamb):
        self.lamb = lamb
        self.w_0 = 0
        self.w = []

    def training(self, x_train, t_train):
        print('\t(using Perceptron method from sklearn)')
        clf = Perceptron(tol=self.lamb, random_state=0)
        clf.fit(x_train, t_train)
        self.w = clf.coef_[0]
        self.w_0 = clf.intercept_

        self.parametres()

    def prediction(self, x):
        result = 0
        pred = self.w_0 + self.w[0] * x[0] + self.w[1] * x[1]
        print("pred =", pred)
        if (pred > 0):
            result = 1

        return result

    @staticmethod
    def error(t, prediction):
        error = 1
        if (t == prediction):
            error = 0

        return error

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w

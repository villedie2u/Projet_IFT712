"""trying to predict targets using the naive bayes method"""

from sklearn.naive_bayes import GaussianNB
from random import shuffle


class Naive_Bayes:
    def __init__(self):
        self.model = GaussianNB()

    def training(self, x_train, t_train):
        # to train the model
        self.model = self.model.fit(x_train, t_train)

    def test_model(self, x_test, t_test):
        # testing the model on test data
        predictions = self.model.predict(x_test)
        nb_good_pred = 0
        n = len(t_test)
        for i in range(n):
            if predictions[i] != -1:
                if t_test[i] == predictions[i]:
                    nb_good_pred += 1
            else:
                n -= 1
        print("\tNumber of true predictions:", nb_good_pred, "/", n, "=", nb_good_pred / n *100, "%")

    @staticmethod
    def init_adaboost(x_train, t_train, n_model):
        # training n models on random samples of data
        indices = list(range(len(x_train))).copy()
        models = []
        n = len(x_train)
        for k in range(n_model):
            shuffle(indices)
            x_training_sample = []
            t_training_sample = []
            for indice in indices[:n//5]:
                x_training_sample.append(x_train[indice])
                t_training_sample.append(t_train[indice])
            models.append(Model(x_training_sample, t_training_sample))
        return models

    @staticmethod
    def ada_boost_predict(x, models):
        predictions = {}
        for model in models:
            prediction = model.predict(x)[0]
            if prediction in predictions.keys():
                predictions[prediction] += 1
            else:
                predictions[prediction] = 1
        # return the prediction the most returned
        pred = None
        n_pred = 0
        for key in predictions.keys():
            if n_pred < predictions[key]:
                n_pred = predictions[key]
                pred = key
        return pred

    def test_ada_boost_model(self, x_train, t_train, n_model, x_test, t_test):
        models = self.init_adaboost(x_train, t_train, n_model)
        ratio = 0
        n = len(x_test)
        for i in range(n):
            if self.ada_boost_predict(x_test[i], models) == t_test[i]:
                ratio += 1
        print("\tNumber of true predictions:", ratio, "/", n, "=", ratio/n*100, "%")


class Model:
    def __init__(self, x_train, t_train):
        self.model = GaussianNB().fit(x_train, t_train)

    def predict(self, x):
        return self.model.predict([x])

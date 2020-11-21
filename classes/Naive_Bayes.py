from sklearn.naive_bayes import GaussianNB


class Naive_Bayes:
    def __init__(self):
        self.model = GaussianNB()

    def training(self, x_train, t_train):
        self.model = self.model.fit(x_train, t_train)

    def test_model(self, x_test, t_test):
        predictions = self.model.predict(x_test)
        nb_good_pred = 0
        n = len(t_test)
        for i in range(n):
            if predictions[i] != -1:
                if t_test[i] == predictions[i]:
                    nb_good_pred += 1
            else:
                n -= 1
        print("\tNombre de prédictions correctes:", nb_good_pred, "/", n, "=", nb_good_pred / n *100, "%")

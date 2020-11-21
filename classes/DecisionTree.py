from sklearn import tree


class DecisionTree:
    def __init__(self):
        self.tree = tree.DecisionTreeClassifier()

    def training(self, x_train, t_train):
        self.tree = self.tree.fit(x_train, t_train)

    def predict(self, x):
        return self.tree.predict([x])

    def test_model(self, x_test, t_test):
        ratio = 0
        n = len(x_test)
        for i in range(n):
            if self.predict(x_test[i]) == t_test[i]:
                ratio += 1
        print("\tNombre de prédictions correctes:", ratio, "/", n, "=", ratio/n, "%")

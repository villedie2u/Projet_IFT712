import tensorflow as tf
import numpy as np


class RN:
    def __init__(self):
        self.model = None

    def training(self, x_train, t_train, n):
        t_train = tf.one_hot(t_train, n)
        # réseau de neurones, ici 3 couches 100 puis n neurones (car n classes)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(n, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["accuracy"]
        )

        # entraînement
        model.fit(x_train, t_train, epochs=200, verbose=0)  # epochs pour le nombre d'entraînements et verbose pour l'affichage
        self.model = model

    def error_predict(self, x_test, t_test, n):
        t_test = tf.one_hot(t_test, n)
        # score sur la base de test
        # la méthode `evaluate` renvoie l'erreur + les
        # métriques qu'on a demandé (ici juste l'accuracy).

        err, acc = self.model.evaluate(x_test, t_test)

    def predict(self, x, t):
        # x doit être un vecteur de 1 élément
        pred = self.model(x)
        print("vrai classe:", t)
        print("prediction:", np.argmax(pred))


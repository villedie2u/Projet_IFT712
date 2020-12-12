"""trying to predict targets using a neural network model"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np


class RN:
    def __init__(self):
        self.model = None
        self.model0 = None

    @staticmethod
    def set_model(nb_layers, nb_neurons, n):
        # creat neural network with selected parameters
        if nb_layers == 3:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dense(n, activation="sigmoid")
            ])
        elif nb_layers == 2:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dense(n, activation="sigmoid")
            ])
        elif nb_layers == 5:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Dense(nb_neurons, activation="relu"),
                tf.keras.layers.Dense(n, activation="sigmoid")
            ])

    def training(self, x_train, t_train, n, nb_layers=3, nb_neurons=100):
        # train the model
        t_train = tf.one_hot(t_train, n)
        # neural network
        model = self.set_model(nb_layers, nb_neurons, n)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["accuracy"]
        )

        # training
        model.fit(x_train, t_train, epochs=200, verbose=0)
        self.model = model

    def error_predict(self, x_test, t_test, n):
        t_test = tf.one_hot(t_test, n)
        # score with test data
        err, acc = self.model.evaluate(x_test, t_test)

    def predict(self, x, t):
        # to predict only 1 sample
        pred = self.model(x)
        print("true target:", t)
        print("prediction:", np.argmax(pred))


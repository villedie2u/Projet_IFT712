""" Main projet_IFT712"""

from classes.reg_log import RegLog
from classes.parser import Parser
from classes.perceptron import Percep
import numpy as np


def main():
    print("\u001B[32m", "============= Main start ===============\n", "\u001B[0m")

    """ using Parser to load the datafile into parser.data"""
    parser = Parser()

    print("\u001B[35m", "\t\t --- reg_log method --- ", "\u001B[0m")
    reg_log = RegLog()
    df_train, df_target = parser.data_sorted_id("data/leaf_train.csv")
    df_target_fusion = parser.get_target_fusion(df_target)

    x_train, x_test, y_train, y_test = parser.get_train_test(df_train, df_target)
    xf_train, xf_test, yf_train, yf_test = parser.get_train_test(df_train, df_target_fusion)

    # pour lancer la régression logistique avec les targets non fusionnées
    print("standard target:")
    reg_log.reg_log(x_train, x_test, y_train, y_test)
    # pour lancer la régression logistique avec les targets fusionnées
    print("merge target:")
    reg_log.reg_log(xf_train, xf_test, yf_train, yf_test)
    print("\u001B[35m", "\t\t --- end reg_log method --- ", "\u001B[0m")

    print("\u001B[35m", "\n\t\t --- perceptron method --- ", "\u001B[0m")

    perceptron = Percep(lamb=0.001)
    df_train, df_target = parser.data_sorted_id("data/leaf_train.csv")
    x_train, x_test, t_train, t_test = parser.get_train_test(df_train, df_target)

    perceptron.training(x_train, t_train)

    # predictions on training and test data
    # predictions_entrainement = np.array([perceptron.prediction(x) for x in x_train])
    # print("Training Error = ", 100 * np.sum(np.abs(predictions_entrainement - t_train)) / len(t_train), "%")

    # predictions_test = np.array([perceptron.prediction(x) for x in x_test])
    # print("Test Error = ", 100 * np.sum(np.abs(predictions_test - t_test)) / len(t_test), "%")

    print("\u001B[35m", "\t\t --- end perceptron method --- ", "\u001B[0m")

    print("\u001B[32m", "\n============= Main end ===============", "\u001B[0m")


if __name__ == "__main__":
    main()

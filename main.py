""" Main projet_IFT712"""

from classes.reg_log import RegLog
from classes.parser import Parser
from classes.perceptron import Percep
from classes.RN import RN
import numpy as np
from classes.KNN import KNNeighbours
from classes.SVM import SupportVectorMachine


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
    knn = KNNeighbours()
    """KNN method with non-merged targets"""
    #knn.Knn(df_train, df_target)
    """KNN method with merged targets"""
    knn.Knn(df_train, df_target_fusion)

    svm = SupportVectorMachine()
    """SVM method with non-merged targets"""
    #svm.Svm(df_train, df_target)
    """SVM method with merged targets"""
    svm.Svm(df_train, df_target_fusion)

    print("\u001B[35m", "\t\t --- end reg_log method --- ", "\u001B[0m")

    print("\u001B[35m", "\n\t\t --- RN method --- ", "\u001B[0m")

    rn = RN()
    df_train, df_target = parser.data_sorted_id("data/leaf_train.csv")
    df_target_fusion = parser.get_target_fusion(df_target)
    x_train, x_test, t_train, t_test = parser.get_train_test(df_train, df_target_fusion)

    n, classes, t_train = parser.modif_target(t_train)
    t_test = parser.modif_target(t_test, classes)[2]
    x_train = parser.modif_entry(x_train)
    x_test = parser.modif_entry(x_test)

    rn.training(x_train, t_train, n)
    rn.error_predict(x_test, t_test, n)

    print("\u001B[35m", "\t\t --- end RN method --- ", "\u001B[0m")

    print("\u001B[32m", "\n============= Main end ===============", "\u001B[0m")


if __name__ == "__main__":
    main()

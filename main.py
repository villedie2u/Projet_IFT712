""" Main projet_IFT712"""

import numpy as np
from classes.parser import Parser
from classes.reg_log import RegLog
from classes.perceptron import Percep
from classes.RN import RN
from classes.KNN import KNNeighbours
from classes.SVM import SupportVectorMachine
from classes.DecisionTree import DecisionTree
from classes.Naive_Bayes import Naive_Bayes


def main():
    print("\u001B[32m", "============= Main start ===============\n", "\u001B[0m")

    """ using Parser to load the datafile into parser.data"""
    parser = Parser()

    df_train, df_target = parser.data_sorted_id("data/leaf_train.csv")
    x_train, x_test, t_train, t_test = parser.get_train_test(df_train, df_target)
    n, classes, t_train = parser.modif_target(t_train)
    t_test = parser.modif_target(t_test, classes)[2]

    df_target_fusion = parser.get_target_fusion(df_target)
    x_train0, x_test0, t_train0, t_test0 = parser.get_train_test(df_train, df_target_fusion)
    n, classes0, t_train0 = parser.modif_target(t_train0, classes=[])
    t_test0 = parser.modif_target(t_test0, classes0)[2]

    print("\u001B[35m", "\t\t --- reg_log method --- ", "\u001B[0m")
    """
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
    
    #KNN method with non-merged targets
    #knn.Knn(df_train, df_target)
    #KNN method with merged targets
    knn.Knn(df_train, df_target_fusion)

    svm = SupportVectorMachine()
    # SVM method with non-merged targets
    #svm.Svm(df_train, df_target)
    # SVM method with merged targets
    svm.Svm(df_train, df_target_fusion)
    """
    print("\u001B[35m", "\t\t --- end reg_log method --- ", "\u001B[0m")

    print("\u001B[35m", "\n\t\t --- RN method --- ", "\u001B[0m")
    """
    rn = RN()
    x_train1 = parser.modif_entry(x_train0)
    x_test1 = parser.modif_entry(x_test0)

    nb_couches = [2, 3, 5]
    nb_neurones = [25, 50, 100]
    for nb_couche in nb_couches:
        for nb_neurone in nb_neurones:
            print(nb_couche, "couches de ", nb_neurone, "neurones")
            rn.training(x_train1, t_train0, n, nb_couche=nb_couche, nb_neurone_par_couche=nb_neurone)
            rn.error_predict(x_test1, t_test0, n)
    """
    print("\u001B[35m", "\t\t --- end RN method --- ", "\u001B[0m")

    print("\u001B[35m", "\t\t --- Decision Tree method --- ", "\u001B[0m")
    """
    dt = DecisionTree()

    print("Résultats sans fusion des classes proches")
    dt.training(x_train, t_train)
    dt.test_model(x_test, t_test)

    print("Résultats avec fusion des classes proches")
    dt.training(x_train0, t_train0)
    dt.test_model(x_test0, t_test0)
    """
    print("\u001B[35m", "\t\t --- end Decision Tree method --- ", "\u001B[0m")

    print("\u001B[35m", "\t\t --- Naive Bayes method --- ", "\u001B[0m")

    nb = Naive_Bayes()

    print("Résultats sans fusion des classes proches")
    nb.training(x_train, t_train)
    nb.test_model(x_test, t_test)

    print("Résultats avec fusion des classes proches")
    nb.training(x_train0, t_train0)
    nb.test_model(x_test0, t_test0)

    print("\u001B[35m", "\t\t --- end Naive Bayes method --- ", "\u001B[0m")

    print("\u001B[32m", "\n============= Main end ===============", "\u001B[0m")


if __name__ == "__main__":
    main()

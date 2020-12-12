""" Main projet_IFT712"""

from classes.parser import Parser
from classes.reg_log import RegLog
from classes.KNN import KNNeighbours
from classes.SVM import SupportVectorMachine
from classes.DecisionTree import DecisionTree
from classes.Naive_Bayes import Naive_Bayes
from classes.RN import RN


def main():
    print("\u001B[32m", "============= Main start ===============\n", "\u001B[0m")

    """ using Parser to load the datafile into parser.data"""
    parser = Parser()

    # data with no merge of targets
    df_train, df_target = parser.data_sorted_id("data/leaf_train.csv")
    x_train, x_test, t_train, t_test = parser.get_train_test(df_train, df_target)
    n, classes, t_train = parser.modif_target(t_train)
    t_test = parser.modif_target(t_test, classes)[2]

    # data with merge of targets
    df_target_fusion = parser.get_target_fusion(df_target)
    x_train0, x_test0, t_train0, t_test0 = parser.get_train_test(df_train, df_target_fusion)
    n, classes0, t_train0 = parser.modif_target(t_train0, classes=[])
    t_test0 = parser.modif_target(t_test0, classes0)[2]

    if int(input("Show Logistic Regression method [yes = 1/ no = 0]: ")):
        print("\u001B[35m", "\t\t --- reg_log method --- ", "\u001B[0m")
        reg_log = RegLog()
        if int(input("Merge targets? [yes = 1/ no = 0]: ")):
            # logistic regression with merged targets
            print("With merge of closest targets")
            reg_log.reg_log(df_train, df_target_fusion)
        else:
            # logistic regression with non-merged targets
            print("Without merge of closest targets")
            reg_log.reg_log(df_train, df_target)

        print("\u001B[35m", "\t\t --- end reg_log method --- \n", "\u001B[0m")

    if int(input("Show KNN method [yes = 1/ no = 0]: ")):
        print("\u001B[35m", "\t\t --- KNN method --- ", "\u001B[0m")
        knn = KNNeighbours()
        if int(input("Merge targets? [yes = 1/ no = 0]: ")):
            # KNN method with merged targets
            print("With merge of closest targets")
            knn.knn(df_train, df_target_fusion)
            print("\u001B[35m", "\t\t --- end KNN method --- \n", "\u001B[0m")
        else:
            # KNN method with non-merged targets
            print("Without merge of closest targets")
            knn.knn(df_train, df_target)

    if int(input("Show SVM method [yes = 1/ no = 0]: ")):
        print("\u001B[35m", "\t\t --- SVM method --- ", "\u001B[0m")
        svm = SupportVectorMachine()
        if int(input("Merge targets? [yes = 1/ no = 0]: ")):
            # SVM method with merged targets
            print("With merge of closest targets")
            svm.Svm(df_train, df_target_fusion)
            print("\u001B[35m", "\t\t --- end SVM method --- \n", "\u001B[0m")
        else:
            # SVM method with non-merged targets
            print("Without merge of closest targets")
            svm.Svm(df_train, df_target)

    if int(input("Show neural network method [yes = 1/ no = 0]: ")):
        print("\u001B[35m", "\n\t\t --- NR method --- ", "\u001B[0m")
        rn = RN()

        if int(input("Merge targets? [yes = 1/ no = 0]: ")):
            print("Results with merge of closest targets")
            x_train1 = parser.modif_entry(x_train0)
            x_test1 = parser.modif_entry(x_test0)
        else:
            print("Results without merge of closest targets")
            x_train1 = parser.modif_entry(x_train)
            x_test1 = parser.modif_entry(x_test)

        layers = [2, 3, 5]
        nb_neurons = [25, 50, 100]
        for layer in layers:
            for neurons in nb_neurons:
                print(layer, "layers of ", neurons, "neurons")
                rn.training(x_train1, t_train0, n, nb_layers=layer, nb_neurons=neurons)
                rn.error_predict(x_test1, t_test0, n)

        print("\u001B[35m", "\t\t --- end NR method --- \n", "\u001B[0m")

    if int(input("Show Decision Tree method [yes = 1/ no = 0]: ")):
        print("\u001B[35m", "\t\t --- Decision Tree method --- ", "\u001B[0m")
        dt = DecisionTree()

        if int(input("Merge targets? [yes = 1/ no = 0]: ")):
            print("Results with merge of closest targets")
            dt.training(x_train0, t_train0)
            dt.test_model(x_test0, t_test0)
        else:
            print("Results without merge of closest targets")
            dt.training(x_train, t_train)
            dt.test_model(x_test, t_test)

        print("\u001B[35m", "\t\t --- end Decision Tree method --- \n", "\u001B[0m")

    if int(input("Show Naive Bayes method [yes = 1/ no = 0]: ")):
        print("\u001B[35m", "\t\t --- Naive Bayes method --- ", "\u001B[0m")
        nb = Naive_Bayes()

        if int(input("Merge targets? [yes = 1/ no = 0]: ")):
            print("Results with merge of closest targets")
            nb.training(x_train0, t_train0)
            nb.test_model(x_test0, t_test0)
        else:
            print("Results without merge of closest targets")
            nb.training(x_train, t_train)
            nb.test_model(x_test, t_test)

        print("\u001B[35m", "\t\t --- end Naive Bayes method --- \n", "\u001B[0m")

    print("\u001B[32m", "\n============= Main end ===============", "\u001B[0m")


if __name__ == "__main__":
    main()

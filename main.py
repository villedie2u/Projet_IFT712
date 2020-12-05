""" Main projet_IFT712"""
from classes.reg_log import RegLog
from classes.parser import Parser
from classes.KNN import KNNeighbours
from classes.SVM import SupportVectorMachine


def main():
    print("\u001B[32m", "============= Main start ===============\n", "\u001B[0m")

    """ using Parser to load the datafile into parser.data"""
    parser = Parser("data/leaf_train.csv")
    # parser.parsing_1()
    # print("> 5 first parameters:", parser.parameters[:5])
    # print("> 15th data: ", parser.data[14])

    print("\u001B[35m", "\t\t --- reg_log method --- ", "\u001B[0m")
    df_train, df_target = parser.data_sorted_id()
    df_target_fusion = parser.get_target_fusion(df_target)
    
    reg_log = RegLog()
    """logistic regression with non-merged targets"""
    reg_log.reg_log(df_train, df_target)
    """logistic regression with merged targets"""
    #reg_log.reg_log(df_train, df_target_fusion)
    
    knn = KNNeighbours()
    """KNN method with non-merged targets"""
    #knn.Knn(df_train, df_target)
    """KNN method with merged targets"""
    #knn.Knn(df_train, df_target_fusion)
    
    svm = SupportVectorMachine()
    """SVM method with non-merged targets"""
    #svm.Svm(df_train, df_target)
    """SVM method with merged targets"""
    #svm.Svm(df_train, df_target_fusion)

    print("\u001B[35m", "\t\t --- end reg_log method --- ", "\u001B[0m")

    print("\u001B[32m", "\n============= Main end ===============", "\u001B[0m")


if __name__ == "__main__":
    main()

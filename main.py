""" Main projet_IFT712"""
from classes.reg_log import RegLog
from classes.parser import Parser


def main():
    print("\u001B[32m", "============= Main start ===============\n", "\u001B[0m")

    """ using Parser to load the datafile into parser.data"""
    parser = Parser("data/leaf_train.csv")
    # parser.parsing_1()
    # print("> 5 first parameters:", parser.parameters[:5])
    # print("> 15th data: ", parser.data[14])

    print("\u001B[35m", "\t\t --- reg_log method --- ", "\u001B[0m")
    reg_log = RegLog()
    df_train, df_target = parser.data_sorted_id()
    df_target_fusion = parser.get_target_fusion(df_target)
    
    """pour lancer la régression logistique avec les targets non fusionnées"""
    reg_log.reg_log(df_train, df_target)
    """pour lancer la régression logistique avec les targets fusionnées """
    reg_log.reg_log(df_train, df_target_fusion)

    print("\u001B[35m", "\t\t --- end reg_log method --- ", "\u001B[0m")

    print("\u001B[32m", "\n============= Main end ===============", "\u001B[0m")


if __name__ == "__main__":
    main()

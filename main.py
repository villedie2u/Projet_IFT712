""" Main projet_IFT712"""
from classes.parser import Parser

def main():
    print("============= Main start ===============\n")

    data = Parser("data/leaf_train.csv")

    print("> les 5 premiers paramètres sont:", data.parameters[:5])
    print("> la 15ième donnée est ", data.data[14])


    print("\n============= Main end ===============")


if __name__ == "__main__":
    main()

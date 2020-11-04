""" to load a csv file content into a python object"""
from classes.csvdataframe import CSVDataFrame
from sklearn.model_selection import train_test_split


class Parser:

    def __init__(self):
        self.data = []
        self.parameters = []

    def parsing_1(self, filename=None):
        filename = filename or self.csv_filename
        print("\u001B[34m", "\tparsing", filename, "...", end='')
        csvfile = open(filename, 'r')
        first_ligne = True
        nb_of_ligne = 0
        for ligne in csvfile:
            if first_ligne:
                first_ligne = False
                self.parameters = ligne.split(",")
            else:
                self.data.append(ligne.split(","))
            nb_of_ligne += 1

        print(" done\n\t\t\t\t\t|", nb_of_ligne - 1, "items loaded \t(in Parser.data)"
                                                     "\n\t\t\t\t\t|", len(self.parameters),
              "parameters \t(in Parser.parameters)"
              "\u001B[0m")

    def data_sorted_id(self, filename):
        """ parsing data into training and testing samples"""
        df = CSVDataFrame(filename).data.sort_values(by=['id'])
        df_train = df.iloc[:, 2:].values
        df_target = df.iloc[:, 1].values
        return df_train, df_target

    @staticmethod
    def get_train_test(df_train,df_target):
        x_train, x_test, y_train, y_test = train_test_split(df_train, df_target, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def get_target_fusion(df_target):
        """modification de df_target avec les noms de classes fusionnées (on ne garde que le préfixe)"""
        df_target_fusion = df_target.copy()
        for i in range(len(df_target_fusion)):
            s = df_target_fusion[i].split('_')
            df_target_fusion[i] = s[0]

        return df_target_fusion

    @staticmethod
    def str_to_float(items):
        float_items = []
        for item in items:
            float_items.append(float(item))
        return float_items

    def get_data_perceptron(self, filename=None):
        self.parsing_1(filename)  # loading data in self.data
        x = []
        t = []
        for item in self.data:
            t.append(item[1])
            x_temp = self.str_to_float(item[2:])
            x.append(x_temp)
        print(len(x), len(t))
        return x, t


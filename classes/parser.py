""" to load a csv file content into a python object"""
from classes.csvdataframe import CSVDataFrame


class Parser:

    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.data = []
        self.parameters = []

    def parsing_1(self):
        print("\u001B[34m", "\tparsing", self.csv_filename, "...", end='')
        csvfile = open(self.csv_filename, 'r')
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

    def data_sorted_id(self):
        """ parsing for the reglog method"""
        df = CSVDataFrame(self.csv_filename).data.sort_values(by=['id'])
        df_train = df.iloc[:, 2:].values
        df_target = df.iloc[:, 1].values
        return df_train, df_target

    def get_target_fusion(self, df_target):
        """modification de df_target avec les noms de classes fusionnées (on ne garde que le préfixe)"""
        df_target_fusion = df_target
        for i in range(len(df_target_fusion)):
            s = df_target_fusion[i].split('_')
            df_target_fusion[i] = s[0]

        return df_target_fusion
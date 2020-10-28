""" to load a csv file content into a python object"""


class Parser:

    def __init__(self, csv_filename):
        print("\u001B[34m", "\tparsing", csv_filename, "...", end='')
        csvfile = open(csv_filename, 'r')
        self.data = []
        self.parameters = []

        first_ligne = True
        nb_of_ligne = 0
        for ligne in csvfile:
            if first_ligne:
                first_ligne = False
                self.parameters = ligne.split(",")
            else:
                self.data.append(ligne.split(","))
            nb_of_ligne += 1

        print(" done\n\t\t\t\t\t|", nb_of_ligne-1, "items loaded \t(in Parser.data)"
              "\n\t\t\t\t\t|", len(self.parameters), "parameters \t(in Parser.parameters)"
              "\u001B[0m")

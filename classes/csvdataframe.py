# -*- coding: utf-8 -*-
"""
to load csv into a dataframe object
"""

import pandas as pd

class CSVDataFrame:
    
    def __init__(self, csv_filename):
        self.data = pd.read_csv(csv_filename)
        
        



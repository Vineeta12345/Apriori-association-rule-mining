# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:23:11 2020

@author: Vineeta
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Association.csv', header = None)
transactions = []
for i in range(0, 20):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 6)])
from apyori import apriori
rules = apriori(transactions,
                min_support = 0.75,
                min_confidence = 0.4,
                min_lift = 1,
                min_length =4)
MB = list(rules)
Result = [list(MB[i][0]) for i in range(0, len(MB))]

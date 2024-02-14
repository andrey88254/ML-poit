#!/bin/env python3

f#rom fxpmath import Fxp
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd
import numpy as np
import math

#import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/VadimKolodny/data/main/sales_data.csv'
s = pd.read_csv("sales_data.csv")
print(s.head(5))


#!/bin/env python3

#from fxpmath import Fxp
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd
import numpy as np
import math

#import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/VadimKolodny/data/main/sales_data.csv'
# s = pd.read_csv(url)
s = pd.read_csv("BostonHousing.csv")
print(s.head(5))


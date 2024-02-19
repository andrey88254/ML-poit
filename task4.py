import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd
import numpy as np
import math

#import matplotlib.pyplot as plt

# s = pd.read_csv(url)

# 120,4,setosa,versicolor,virginica
# 0=setosa,1=versicolor,2=virginica

s = pd.read_csv("iris_training.csv")
print(s.head(5))
print()
print(s.describe())
print()

setosa_df=s.loc[s["Species"]==0].copy()
versicolor_df=s.loc[s["Species"]==1].copy()
virginica_df=s.loc[s["Species"]==2].copy()

print("setosa")
print(setosa_df.describe())
print()
print("versicolor")
print(versicolor_df.describe())
print()
print("virginica")
print(virginica_df.describe())

print()
print("Средняя длина чашестика")

d={"setosa" : setosa_df["SepalLength"].mean(),
   "versicolor" : versicolor_df["SepalLength"].mean(),
   "virginica" : virginica_df["SepalLength"].mean()}

print(d)
import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd
import numpy as np

df1 = pd.read_csv("df1.csv")
df2 = pd.read_csv("df2.csv")
print("df1")
print(df1)
print()
print("df2")
print(df2)
output_df = pd.merge(df1,df2,
                    on="nom",
                    how="outer")
print()
print("Объединение по столбцу \"nom\" :")
print(output_df)
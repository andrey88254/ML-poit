import warnings
#warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import numpy as np

mtr = np.random.randint(-10,10,(4,4))
vect = np.random.randint(-10,10,4)
print("Матрица:")
print(mtr)
print()
print("Вектор :")
print(vect)
print()
print("Результат :")
result = np.dot(mtr,vect)
print(result)
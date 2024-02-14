from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # поддержка многомерных массивов; поддержка высокоуровневых математических функций, предназначенных для работы с многомерными массивами
import pandas as pd # Предоставляет специальные структуры данных и операции для манипулирования числовыми таблицами и временны́ми рядами
import matplotlib.pyplot as plt # представляет собой набор функций, благодаря которым matplotlib работает как MATLAB.
#from IPython.display import clear_output
from six.moves import urllib
#from google.colab import drive
#
import tensorflow as tf
from sklearn.datasets import load_boston
#
import tensorflow.feature_column as fc


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # внутренняя функция, её значение будет возвращено
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # создать объект tf.data.Dataset с данными и его меткой
        if shuffle:
            ds = ds.shuffle(1000)  # упорядочить набор данных случайным образом
        ds = ds.batch(batch_size).repeat(num_epochs)  # разделить набор данных на пакеты по 32 и повторить процесс для количества эпох
        return ds  # возвращаем пакет набора данных
    return input_function  # возвращаем объект функции для использования

# Загрузить набор данных.
#drive.mount('/content/drive')
#dfeval = pd.read_csv(
##    parse_dates=['Date'])
#!head /content/drive/MyDrive/data/titanic.csv
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # данные для тренировок
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # данные для испытаний
print(dftrain.keys())
#dfeval = pd.read_csv('https://drive.google.com/uc?export=download&id=1ZGacse9wHc6Pm1-HswoY90rHJyZENCCY')
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")
# print(dftrain.describe())
# print(dftrain.count())
# print(dftrain.head(10))

# print(y_train.head())
# print(y_train)
# print(dftrain.sex.value_counts())
#dftrain.age.hist(bins=20)
# plt.hist(dftrain.age)
#
# plt.show()
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # получает список всех уникальных значений из заданного столбца признаков
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#print(feature_columns)
#
train_input_fn = make_input_fn(dftrain, y_train)  # здесь мы вызовем функцию input_function, которая была возвращена нам, чтобы получить объект набора данных, который мы можем передать модели.
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)  # тренировка
result = linear_est.evaluate(eval_input_fn)  # получить метрики/статистику модели путем тестирования тестовых данных
# clear_output()  # очистить выходную консоль
print()
print(result['accuracy'])  # переменная которая хранит результат — это просто набор статистических данных о нашей модели.

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
print(probs)

plt.hist(probs, bins=20, edgecolor="black")
plt.title("предсказание вероятности")


# model_vars=dict(linear_est.get_variable_names(), linear_est.get_variable_value())
print("coef")
model_coef_names=linear_est.get_variable_names()
print(model_coef_names)
for c_name in model_coef_names:
    print(c_name)

print()
for c_name in model_coef_names:
    print(c_name)
    print(linear_est.get_variable_value(c_name))
    print()

plt.show()

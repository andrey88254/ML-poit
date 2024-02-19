import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist  # загружаем набор данных

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # разделяем на данные для тестирования и обучения
print(train_images.shape)
# print(train_images[0,23,23])
# print(train_labels[:20])
#
# plt.figure()
# plt.imshow(test_images[999])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0
#print(test_images[999])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # входной слой (1)
    keras.layers.Dense(5, activation='relu'),  # скрытый слой (2)  # скрытый слой
    keras.layers.Dense(10, activation='softmax') # выходной слой (3)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

print()
print("Точность на тестовом наборе:")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('Test accuracy:', test_acc)
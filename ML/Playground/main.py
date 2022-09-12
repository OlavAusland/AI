import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from matplotlib import pyplot as plt

num_classes = 3
input_shape = (32, 32, 1)
# 0 = left, 1 = right
def load_data():
    train_data = []
    train_labels = []

    for file in os.listdir('hand-gesture-data/left'):
        img = Image.open(f'hand-gesture-data/left/{file}')
        img = img.resize((input_shape[0], input_shape[1]))
        img = np.asarray(img, dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        train_data.append(img / 255.0)
        train_labels.append(0)

    for file in os.listdir('hand-gesture-data/right'):
        img = Image.open(f'hand-gesture-data/right/{file}')
        img = img.resize((input_shape[0], input_shape[1]))
        img = np.asarray(img, dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        train_data.append(img / 255.0)
        train_labels.append(1)


    train_data = np.expand_dims(train_data, -1)
    train_labels = keras.utils.to_categorical(train_labels, num_classes)

    return train_data, train_labels


def main():
    train_x, train_y = load_data()
    batch_size = 128
    epochs = 15

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ]
    )

    model.summary()
    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_x, train_y, batch_size, epochs, validation_split=0.2)
    model.predict(train_x)
    model.save('./model.h5')


if __name__ == '__main__':
    main()

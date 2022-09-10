import os

import cv2
from PIL import Image
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

input_shape = (32, 32, 1)
num_classes = 2

def load_data():
    train_data = []
    train_labels = []
    for file in os.listdir('./Data/Left'):
        img = Image.open(f'./Data/Left/{file}')
        img = img.resize((input_shape[0], input_shape[1]))
        img = np.asarray(img, dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        train_data.append(img / 255.0)
        train_labels.append(0)

    for file in os.listdir('./Data/Right'):
        img = Image.open(f'./Data/Right/{file}')
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
    model = keras.models.load_model('./model.h5')
    print(model.predict(train_x[0]))


if __name__ == '__main__':
    main()
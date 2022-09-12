import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sys
np.set_printoptions(threshold=sys.maxsize)

num_classes = 10
input_shape = (28, 28, 1)
(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
print(test_labels)
# SCALE DATA BETWEEN 0 & 1
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

print(type(train_data))
print(train_data.shape)
train_data = np.expand_dims(train_data, -1)
test_data = np.expand_dims(test_data, -1)

train_labels = keras.utils.to_categorical(train_labels, num_classes)


test_labels = keras.utils.to_categorical(test_labels, num_classes)

# CREATE THE MODEL
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

# TRAIN SPECS
batch_size = 128
epochs = 15

model.compile(
    "adam",
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.predict(test_data)
model.save('./model.h5')

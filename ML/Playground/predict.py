import os

import cv2
from PIL import Image
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

input_shape = (32, 32, 1)
num_classes = 2


def main():
    image = None
    cap = cv2.VideoCapture(0)
    model = keras.models.load_model('model.h5')

    pose_map = {0:'Left', 1:'Right'}

    while cap.isOpened():
        _, frame = cap.read()
        pred_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pred_frame = cv2.resize(pred_frame, dsize=(32, 32))
        pred_frame = np.reshape(pred_frame, input_shape)
        pred_frame = np.asarray(pred_frame, dtype=np.float32) / 255.0
        face_prediction = model.predict(np.expand_dims(pred_frame, axis=0))
        print(face_prediction)
        cv2.putText(img=frame, text=pose_map[face_prediction[0].argmax()], org=(20, 20), color=[255,255,255],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)


        cv2.imshow('feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
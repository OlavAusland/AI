import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('./model.h5')
drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global img
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)
    elif event==cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512, 512, 1), np.uint8)


img = np.zeros((512,512,1), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',img)
    img.resize((28, 28, 1), refcheck=False)
    print(model.predict(np.expand_dims(img, axis=0)))
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
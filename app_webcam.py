import cv2
import streamlit as st
import tensorflow as tf

import joblib
from PIL import Image
from skimage.transform import resize
import numpy as np
import time


def predict(image):

    Y_prediction = CNN.predict(image)


    return Y_prediction.argmax()

CNN = tf.keras.models.load_model('Models/hand_numbers')

st.title("Clasificador de numeros en señas")
run = st.checkbox('Conectar')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image = np.asarray(frame)/255
    my_image= resize(image, (64,64)).reshape((1, 64,64,3))
    prediction = predict(my_image)
    cv2.putText(frame, str(prediction), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 4, 255,5)
    FRAME_WINDOW.image(frame)
    

else:
    st.write('Detenido')

 
import tensorflow as tf
import streamlit as st
import joblib
from PIL import Image
from skimage.transform import resize
import numpy as np
import time

#loading the cat classifier model
CNN = tf.keras.models.load_model('Models/hand_numbers')


def predict(image):
    
    Y_prediction = CNN.predict(image)
    
         
    return Y_prediction.argmax()

# Designing the interface
st.title("Clasificador de numeros manuales")
# For newline
st.write('\n')

image = Image.open('images/image.jpg')
show = st.image(image, use_column_width=True)

st.sidebar.title("Subir imagen")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader("Programa elaborado para Trascender Global, cuenta con un modelo desarrollado en Keras, el repositorio incluye el notebook utilizado en el entrenamiento del modelo y ademas con los modulos utilizados en el procesamiento de los datos y la construccion del modelo. El modelo actual cuenta con 90% de precision",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Imagen actual', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    image = np.asarray(u_img)/255
    
    my_image= resize(image, (64,64)).reshape((1, 64,64,3))

# For newline
st.sidebar.write('\n')
    
if st.sidebar.button("Click aquí para clasificar"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Porfavor sube una imagen")
    
    else:
        
        with st.spinner('Clasificando numero ...'):
            
            prediction = predict(my_image)
            time.sleep(2)
            st.success('Listo!')
            
        st.sidebar.header("Predicción: ")            
        st.sidebar.write("Es el número : ", prediction)
import tensorflow as tf
from tensorflow.keras.layers import * 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import L2
from tensorflow.keras.applications.resnet50 import  preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(
    page_title="CatVsDog",
    layout='wide',
)
@st.cache_resource
def build_model():
    resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(128,128,3))
    model = Sequential([
        resnet,
        GlobalAveragePooling2D(),
        Dense(32,activation='relu'),
        BatchNormalization(),
        Dropout(.2),
        
        Dense(32,activation='relu',kernel_regularizer=L2(0.001)),
        BatchNormalization(),
        Dropout(.2),
        
        Dense(16,activation='relu',kernel_regularizer=L2(0.001)),
        BatchNormalization(),
        Dropout(.2),
        
        Dense(1,activation='sigmoid',kernel_regularizer=L2(0.001)),  
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.load_weights(r'modv2.weights.h5')
    return model
model = build_model()

def preprocess(img):
    img = plt.imread(img)
    ig = img.copy()
    ig = ig[:,:,:3]
    ig = tf.keras.preprocessing.image.smart_resize(ig,size=(128,128))
    if ig.max() < 2:
        ig = ig*255
    ig = np.array(ig)
    ig = np.expand_dims(ig,axis=0)
    ig = preprocess_input(ig)
    return ig[0],img
    


img = st.sidebar.file_uploader("Upload your image here")
btn = st.sidebar.button("Predict")
if btn and img:
    temp,img2 = preprocess(img)
    pred = model.predict(np.array([temp]))
    st.title("Dog" if np.round(pred)[0] == 1 else "Cat")
    st.image(img2 / img2.max())
else :
    st.title("CatVsDog")
    st.title('Enter your photo to predict')



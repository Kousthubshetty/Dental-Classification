import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import keras
from tensorflow import keras
from tensorflow.python.keras.models import *
# from tensorflow.keras import preprocessing
from keras import preprocessing
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout,Embedding,LSTM,GRU
from keras.preprocessing.image import ImageDataGenerator
# from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from keras.optimizers import Adam
# from keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


fig = plt.figure()
st.title('Dental Curie detection')
st.markdown("Prediction : (Dental Problems)")

def predict(image):
    classifier_model = 'VGG16-Model.h5'
    model=keras.models.load_model(classifier_model)

    from keras import applications
    vgg16 = applications.VGG16(include_top=False, weights='imagenet')
    test_image = image.resize((224, 224))
    test_image = np.array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = {0: 'CAVITY', 1: 'COLD SORES',2:"DEAD TOOTH",3:"GINGIVITY",4:"HEALTHY"}
    bt_prediction = vgg16.predict(test_image)
    preds = model.predict(bt_prediction)
    labels = ['Cavity','Cold','Dead','Gingivitis','Healthy']
    for idx, animal, x in zip(range(0,6), labels , preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2)))
    predictions=preds
    print(predictions)
    a=np.array(max(predictions))
    print(class_names[np.argmax(a)])
    result = f"Predicted as {class_names[np.argmax(a)]} with {(max(a)*100).__round__(2)} % confidence. "

    return result


file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
class_btn = st.button("CHECK")
if file_uploaded is not None:
    image = Image.open(file_uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=True)

if class_btn:
    if file_uploaded is None:
        st.write("Invalid command, please upload an image")
    else:
        with st.spinner('Model working....'):
            plt.imshow(image)
            plt.axis("off")
            predictions = predict(image)
            time.sleep(1)
            st.success('Prediction Succesful.')
            st.write(predictions)




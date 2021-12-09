import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_DIR = os.path.join(os.path.dirname('C:/Users/edenl/Desktop/ia_coding/deep_learning/RNN , CNN/CNN'), 'model.h5')

st.set_option('depreciation.showileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('/content/my_model2.hdf5')
    return model
model=load_model()
st.write("""
        # Number Classification
         """)

file = st.file_uploader("Please upload an number image", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):

    size = (180,180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    predciction = model.predict(img_reshape)

    return import_and_predict
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    string="This image most likely is: "+class_names[np.argmax(predictions)]
    st.success(string)
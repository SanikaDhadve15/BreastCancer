import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from keras import backend as K

# Define the model architecture
class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = tf.keras.models.Sequential()
        shape = (height, width, depth)
        channelDim = -1
        if K.image_data_format() == "channels_first":
            shape = (depth, height, width)
            channelDim = 1

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(axis=channelDim))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(classes, activation='softmax'))
        return model

# Load the trained model
def load_model():
    model = CancerNet.build(width=48, height=48, depth=3, classes=2)
    try:
        model.load_weights("C:/Users/Hp/OneDrive/Desktop/Project1Inter/breastcancer.h5")
        st.success("Model loaded successfully!")
    except Exception as e:
        pass
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (48, 48))  # Resize to match model input
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Breast Cancer Detection")
st.write("Upload an image to predict whether it is malignant or benign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    result = "Malignant (Cancerous)" if prediction[0][0] > 0.5 else "Benign (Non-Cancerous)"
    
    st.write(f"**Prediction:** {result}")

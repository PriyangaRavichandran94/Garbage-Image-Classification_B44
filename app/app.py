import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle

model = load_model("models/model.keras")

with open("models/classes.pkl", "rb") as f:
    classes = pickle.load(f)

classes = list(classes.keys())

st.title("♻️ Garbage Classifier")

file = st.file_uploader("Upload Image")

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

if file:
    image = Image.open(file).convert("RGB")   # ✅ fix RGBA issue too
    st.image(image)

    img = preprocess(image)
    pred = model.predict(img)[0]

    # 🔹 Single Prediction
    st.subheader("Prediction:")
    st.write(f"{classes[np.argmax(pred)]} ({max(pred)*100:.2f}%)")

    # 🔹 Top 3 Predictions
    st.subheader("Top Predictions:")
    top3 = np.argsort(pred)[-3:][::-1]

    for i in top3:
        st.write(f"{classes[i]}: {pred[i]*100:.2f}%")
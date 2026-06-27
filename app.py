import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Garbage Classification System",
    page_icon="♻️",
    layout="wide"
)

# ==========================================
# Load Model
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_model.h5")


@st.cache_data
def load_classes():
    with open("models/class_names.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
class_names = load_classes()

# Dataset path
DATA_DIR = "data/raw"

# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("♻️ Garbage Classification")

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Dataset Analysis (EDA)"]
)

# ============================================================
# Prediction Page
# ============================================================

if page == "Prediction":

    st.title("♻️ Garbage Classification System")

    st.write(
        "Upload an image to classify garbage into categories such as "
        "plastic, paper, glass, metal, cardboard, and trash."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((224, 224))

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)

            prediction_percent = prediction[0] * 100

            top3 = np.argsort(prediction_percent)[-3:][::-1]

            predicted_class = class_names[top3[0]]
            confidence = prediction_percent[top3[0]]

            st.success(f"✅ Prediction : **{predicted_class}**")

            st.info(f"🎯 Confidence : **{confidence:.2f}%**")

            st.progress(int(confidence))

            st.subheader("🔝 Top 3 Predictions")

            for idx in top3:
                st.write(
                    f"**{class_names[idx]}** : "
                    f"{prediction_percent[idx]:.2f}%"
                )

            # Probability Chart
            st.subheader("📊 Prediction Probability")

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.barh(class_names, prediction_percent)

            ax.set_xlabel("Probability (%)")

            ax.set_title("Prediction Probability")

            st.pyplot(fig)

        except Exception as e:

            st.error(f"Error : {e}")

# ============================================================
# EDA PAGE
# ============================================================

elif page == "Dataset Analysis (EDA)":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "This section provides insights into the garbage dataset "
        "used for training the deep learning model."
    )

    classes = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    # ==========================================
    # Dataset Summary
    # ==========================================

    st.header("📁 Dataset Summary")

    total_images = 0
    counts = []

    for cls in classes:

        class_path = os.path.join(DATA_DIR, cls)

        count = len([
            img for img in os.listdir(class_path)
            if img.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        counts.append(count)

        total_images += count

    col1, col2 = st.columns(2)

    col1.metric("Total Classes", len(classes))

    col2.metric("Total Images", total_images)

    # ==========================================
    # Images per class
    # ==========================================

    st.header("📊 Number of Images per Class")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(classes, counts)

    ax.set_xlabel("Garbage Category")

    ax.set_ylabel("Number of Images")

    ax.set_title("Images per Class")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ==========================================
    # Class Count Table
    # ==========================================

    st.header("📋 Class Distribution")

    st.table({
        "Class": classes,
        "Images": counts
    })

    # ==========================================
    # Sample Images
    # ==========================================

    st.header("🖼 Sample Images")

    cols = st.columns(3)

    for i, cls in enumerate(classes):

        class_path = os.path.join(DATA_DIR, cls)

        img_name = next(
            img for img in os.listdir(class_path)
            if img.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        )

        img = Image.open(
            os.path.join(class_path, img_name)
        )

        with cols[i % 3]:
            st.image(
                img,
                caption=cls,
                use_container_width=True
            )

    # ==========================================
    # Model Information
    # ==========================================

    st.header("🤖 Model Information")

    st.markdown("""
**Model Used:** MobileNetV2

**Transfer Learning:** Yes

**Input Image Size:** 224 × 224

**Optimizer:** Adam

**Loss Function:** Sparse Categorical Crossentropy

**Evaluation Metric:** Accuracy

**Epochs:** 10

**Framework:** TensorFlow + Streamlit
""")
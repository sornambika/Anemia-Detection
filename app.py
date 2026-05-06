import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras_cv_attention_models import convnext
from tensorflow.keras.applications.convnext import preprocess_input

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="Anemia Detection",
    page_icon="🩸",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("anemia_convnext_model.keras/")
    return model

model = load_model()

# ---------------- IMAGE PREPROCESSING ---------------- #

IMG_SIZE = 224

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image).astype("float32")

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image

# ---------------- PREDICTION FUNCTION ---------------- #

def predict(image):

    processed = preprocess_image(image)

    prediction = model.predict(processed)

    st.write("Raw Model Output:", prediction)

    # Case 1: Sigmoid output
    if prediction.shape[1] == 1:

        prob = prediction[0][0]

        if prob > 0.5:
            return "Non Anemic", float(prob)
        else:
            return "Anemia Detected", float(1 - prob)

    # Case 2: Softmax output
    else:

        probs = prediction[0]
        class_id = np.argmax(probs)

        if class_id == 0:
            return "Anemia Detected", float(probs[0])
        else:
            return "Non Anemic", float(probs[1])

# ---------------- UI ---------------- #

st.title("🩸 AI Based Anemia Detection System")

st.markdown(
"""
Detect **Anemia from Eye Conjunctiva Images** using a deep learning model based on **ConvNeXt Architecture**.
"""
)

st.divider()

col1, col2 = st.columns(2)

# ---------------- LEFT COLUMN ---------------- #

with col1:

    st.subheader("📤 Upload Eye Image")

    uploaded_file = st.file_uploader(
        "Upload Eye Conjunctiva Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", width=300)

        if st.button("🔍 Analyze Image"):

            with st.spinner("Analyzing image using AI model..."):

                result, confidence = predict(image)

            st.divider()

            st.subheader("🧠 Prediction Result")

            if result == "Anemia Detected":
                st.error("⚠️ Anemia Detected")
            else:
                st.success("✅ Non Anemic")

            st.write("### Confidence Score")

            st.progress(confidence)

            st.write(f"Confidence: **{confidence:.2f}**")

# ---------------- RIGHT COLUMN ---------------- #

with col2:

    st.subheader("ℹ️ About This System")

    st.info(
        """
        This AI-powered system detects **anemia using eye conjunctiva images**.

        **Model Used**
        - ConvNeXt Deep Learning Model

        **Input**
        - Eye Conjunctiva Image

        **Output**
        - Anemia Detected
        - Non Anemic

        This system helps in **non-invasive preliminary anemia screening**.
        """
    )

    st.subheader("📌 Instructions")

    st.write(
        """
        1️⃣ Upload a **clear eye conjunctiva image**  
        2️⃣ Click **Analyze Image**  
        3️⃣ AI model predicts anemia condition  
        4️⃣ View prediction and confidence score
        """
    )

    # st.subheader("⚠️ Disclaimer")

    # st.warning(
    #     """
    #     This tool is for **educational and research purposes only**.

    #     It should **not replace professional medical diagnosis**.
    #     """
    # )

st.divider()

st.caption("Developed for AI-based Anemia Detection Project")
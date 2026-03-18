import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# PAGE CONFIG (IMPORTANT 🔥)
# -----------------------------
st.set_page_config(page_title="Food Calorie Estimator", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("food_model_advanced.keras")

class_names = ['apple', 'banana', 'burger', 'donut', 'fries', 'pizza']

calories = {
    "pizza": 300,
    "burger": 250,
    "apple": 80,
    "banana": 100,
    "fries": 350,
    "donut": 200
}

# -----------------------------
# SIDEBAR (PRO LOOK)
# -----------------------------
st.sidebar.title("🍔 App Info")

st.sidebar.write("Deep Learning Food Classifier")
st.sidebar.write("Model: MobileNetV2")

if st.sidebar.button("Reset Calories"):
    st.session_state.total_calories = 0
    st.rerun()

# -----------------------------
# SESSION STATE
# -----------------------------
if "total_calories" not in st.session_state:
    st.session_state.total_calories = 0

# -----------------------------
# HEADER
# -----------------------------
st.title("🍔 Food Calorie Estimator")
st.markdown("### Upload food images and track your calorie intake")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Food Image(s)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# -----------------------------
# MAIN LAYOUT (2 COLUMNS)
# -----------------------------
if uploaded_files:

    col1, col2 = st.columns(2)

    for uploaded_file in uploaded_files:

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # -----------------------------
        # PREPROCESS
        # -----------------------------
        img = image.resize((224, 224))
        img = np.array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        with st.spinner("Analyzing..."):
            prediction = model.predict(img)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # -----------------------------
        # RIGHT COLUMN (RESULTS)
        # -----------------------------
        with col2:
            st.subheader("🔍 Results")

            st.success(f"🍽 {predicted_class.capitalize()}")
            st.metric("🔥 Calories", f"{calories[predicted_class]} kcal")
            st.metric("📊 Confidence", f"{confidence:.2f}%")

            st.progress(int(confidence))

            # Top-2 predictions
            st.write("Top Predictions:")
            top_indices = prediction[0].argsort()[-2:][::-1]
            for i in top_indices:
                st.write(f"{class_names[i]} → {prediction[0][i]*100:.2f}%")

            # Category
            if calories[predicted_class] < 100:
                st.info("Low Calorie")
            elif calories[predicted_class] < 250:
                st.warning("Moderate Calorie")
            else:
                st.error("High Calorie")

            # Quantity
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                max_value=10,
                value=1,
                key=uploaded_file.name
            )

            if st.button("Add to Daily Intake", key=uploaded_file.name + "_btn"):
                total = calories[predicted_class] * quantity
                st.session_state.total_calories += total
                st.success(f"Added {total} kcal")

# -----------------------------
# FOOTER DASHBOARD
# -----------------------------
st.markdown("---")
st.subheader("📅 Daily Tracker")

st.metric("Total Calories Today", f"{st.session_state.total_calories} kcal")

st.info("Powered by MobileNetV2 (Transfer Learning)")
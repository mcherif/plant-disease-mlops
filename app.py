import streamlit as st
from PIL import Image
from src.model import load_model, preprocess_image, predict_image

# Display logo and title side by side
col1, col2 = st.columns([1, 3])
with col1:
    st.image(
        "https://huggingface.co/spaces/mcherif/plant-disease-api/resolve/main/images/plant-disease-logo.png",
        width=120,
    )
with col2:
    st.title("Plant Disease Classifier")
    st.write("Upload a plant leaf image to get disease prediction.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Load model (cache to avoid reloading on every run)
    @st.cache_resource
    def get_model():
        return load_model()

    model, processor, idx_to_class, device = get_model()

    # Preprocess the image for the model
    img_tensor = preprocess_image(image, processor, device)

    # Make prediction
    prediction = predict_image(model, img_tensor, idx_to_class)

    st.write(f"**Prediction:** {prediction['class']}")
    st.write(f"**Confidence:** {prediction['confidence']:.2f}")

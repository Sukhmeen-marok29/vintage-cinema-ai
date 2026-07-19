import streamlit as st
from PIL import Image
import os
from inference import process_pipeline

st.set_page_config(page_title="Vintage Cinema AI", layout="wide")

st.title("🎬 Vintage Cinematic Aesthetic Classifier & Colorist")
st.write("Upload a modern photograph. The AI will classify its underlying composition style and use a Neural Style Transfer backbone to transform it into an era-specific cinematic look.")

uploaded_file = st.file_uploader("Choose a modern image...", type=["jpg", "jpeg", "png"])

# Layout setup
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Save file locally to work across components
    temp_input_path = "temp_input.jpg"
    image = Image.open(uploaded_file)
    image.save(temp_input_path)
    
    with col1:
        st.subheader("📸 Original Input Photo")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("🎞️ Transformed Cinematic Output")
        with st.spinner("Analyzing style properties and computing color mapping tensors..."):
            try:
                # Execute pipeline execution
                era_label, optimized_array = process_pipeline(temp_input_path)
                
                # Render results out cleanly
                st.success(f"Detected Base Style: **{era_label.replace('_', ' ').title()}**")
                st.image(optimized_array, caption="Processed Aesthetic Output", use_container_width=True)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
                st.info("Tip: Ensure you have placed reference files under dataset/styles/ directory.")

    # Cleanup local footprint
    if os.path.exists(temp_input_path):
        os.remove(temp_input_path)
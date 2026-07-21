import streamlit as st
from PIL import Image
import os
from inference import process_pipeline

st.set_page_config(layout="wide", page_title="Vintage Cinema AI")

st.title("🎬 Vintage Cinematic Aesthetic Classifier & Colorist")
st.write("Upload an image to automatically classify its photographic structure and map historic movie grading styles onto it.")

# 🎛️ Sidebar Controls
st.sidebar.header("🕹️ Engine Configurations")

mode = st.sidebar.radio(
    "Processing Pipeline Mode:",
    ("Instant Color Mode", "Deep Neural Optimization (Slow CPU)")
)

era_override = st.sidebar.selectbox(
    "Target Style Era:",
    ("Auto-Detect", "70s_technicolor", "90s_filmic", "neo_noir")
)

intensity = st.sidebar.slider(
    "Vintage Grading Intensity:",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1
)

uploaded_file = st.file_uploader("Choose a modern image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp input asset
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Input Photo")
        st.image(temp_path, use_container_width=True)
        
    with col2:
        st.subheader("🎞️ Transformed Cinematic Output")
        
        with st.spinner("Processing image textures..."):
            pred_era, final_era, output_array = process_pipeline(
                temp_path, 
                selected_era=era_override, 
                mode=mode, 
                intensity=intensity
            )
            
            # Show processing metrics
            st.success(f"🤖 AI Detected Base Layout: **{pred_era.upper()}**")
            if era_override != "Auto-Detect":
                st.warning(f"🎨 Applied User Override: **{final_era.upper()}**")
                
            st.image(output_array, use_container_width=True)
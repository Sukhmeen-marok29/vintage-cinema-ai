# 🎬 Vintage Cinematic Aesthetic Classifier & Image Colorist

An intelligent computer vision and generative engine that reads modern digital photographs, auto-classifies their underlying visual era, and executes an architecture-specific Neural Style Transfer pipeline to map vintage visual grades onto modern inputs.

## 🚀 Tech Stack
* **Core Machine Learning:** PyTorch, Torchvision
* **Image Processing:** OpenCV, Pillow
* **Front-End Deployment:** Streamlit

## 📂 Execution Layout
1. Train the baseline weights: `python train_classifier.py`
2. Spin up the front-end web portal: `streamlit run app.py`
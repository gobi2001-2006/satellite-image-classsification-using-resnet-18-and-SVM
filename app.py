import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import cv2
import numpy as np
from skimage.feature import hog
import joblib

# -------------------- DARK GRADIENT STYLE SETUP --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    }

    h4 {
        color: #ffa500;
    }

    .result-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .floating-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #FF6347;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }

    .floating-btn:hover {
        transform: scale(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD MODELS --------------------
try:
    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 10)
    cnn_model.load_state_dict(torch.load('resnet18_eurosat_model.pth', map_location=torch.device('cpu')))
    cnn_model.eval()

    svm, scaler, le = joblib.load("svm_model_with_scaler_and_encoder.joblib")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -------------------- FEATURE EXTRACTION FOR ML --------------------
def extract_ml_features(image_bgr):
    image_resized = cv2.resize(image_bgr, (64, 64))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)

    chans = cv2.split(image_resized)
    hist_features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
        hist_features.extend(hist.flatten())

    return np.hstack([hog_feat, hist_features]).reshape(1, -1)

# -------------------- UI TITLE --------------------
st.title("🛰 Satellite Image Classification")
st.markdown("Upload an image to predict its class and view the confidence score.\n\n**Choose a satellite image**")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((400, 400))
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # --- CNN prediction ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        cnn_output = cnn_model(tensor)
        cnn_pred = torch.argmax(cnn_output, dim=1)
        cnn_conf = torch.nn.functional.softmax(cnn_output, dim=1)[0][cnn_pred].item()

    # --- ML (SVM) prediction ---
    features = extract_ml_features(image_bgr)
    features_scaled = scaler.transform(features)
    ml_pred = svm.predict(features_scaled)[0]
    ml_conf = np.max(svm.predict_proba(features_scaled))

    classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

    cnn_class = classes[cnn_pred.item()]
    ml_class = le.inverse_transform([ml_pred])[0]

    # --- Layout with side-by-side view ---
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(image_resized, caption="📷 Uploaded Satellite Image", use_column_width=True)

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>🧠 CNN Prediction: {cnn_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>Confidence: {cnn_conf:.2%}</h4>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>🔍 ML (SVM) Prediction: {ml_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>Confidence: {ml_conf:.2%}</h4>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FLOATING BUTTON --------------------
st.markdown('<button class="floating-btn">+</button>', unsafe_allow_html=True)

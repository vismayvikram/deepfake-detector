# app.py
# small streamlit app for testing the detector prototype

import os
import torch
import torch_dct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import torchvision.transforms as T

from model import Detector

st.set_page_config(page_title="AI Face Detector Prototype", layout="centered")

st.title("Synthetic Face Detector (Prototype)")
st.write(
    "A research prototype testing if combining spatial features (ResNet-18) with "
    "DCT frequency-domain features catches AI-generated faces better than spatial alone."
)

st.warning(
    "**Important limitation:** This prototype classifies single static photos. "
    "It is NOT a video deepfake detector — it cannot evaluate temporal tells like "
    "blinking irregularities or lip-sync drift, which is a completely different problem."
)

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # prefer fusion_aug if trained, otherwise fusion
    model_path = "checkpoints/fusion_aug.pt"
    if not os.path.exists(model_path):
        model_path = "checkpoints/fusion.pt"
        
    if not os.path.exists(model_path):
        return None, device, "No checkpoint found in checkpoints/"
        
    model = Detector(mode="fusion")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device, model_path

model, device, ckpt_info = load_model()

if model is None:
    st.error(f"Could not load detector: {ckpt_info}. Please run train.py first!")
else:
    st.success(f"Loaded checkpoint: `{ckpt_info}` on `{device}`")

uploaded_file = st.file_uploader("Upload a face image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
    # compute prediction
    if model is not None:
        # transform
        img_resized = img.resize((256, 256), Image.BILINEAR)
        img_t = T.ToTensor()(img_resized).unsqueeze(0).to(device)
        
        with torch.no_grad():
            main_out, s_out, f_out = model(img_t)
            prob_fake = torch.sigmoid(main_out).item()
            prob_spatial = torch.sigmoid(s_out).item() if s_out is not None else 0.5
            prob_freq = torch.sigmoid(f_out).item() if f_out is not None else 0.5
            
        is_fake = prob_fake >= 0.5
        conf = prob_fake if is_fake else (1.0 - prob_fake)
        label = "Synthetic / AI-Generated" if is_fake else "Real Photograph"
        
        with col2:
            st.subheader("Classification Result")
            if is_fake:
                st.error(f"**{label}**")
            else:
                st.success(f"**{label}**")
                
            st.metric("Overall Fake Probability", f"{prob_fake * 100:.1f}%")
            st.write(f"Confidence: **{conf * 100:.1f}%**")
            
            st.write("---")
            st.caption("Branch breakdown:")
            st.write(f"- Spatial branch vote: **{prob_spatial * 100:.1f}% fake**")
            st.write(f"- Frequency branch vote: **{prob_freq * 100:.1f}% fake**")
            
    # DCT spectrum visualization
    st.write("---")
    st.subheader("2D Discrete Cosine Transform (DCT) Spectrum")
    
    # compute DCT
    gray = img.convert("L").resize((256, 256))
    gray_t = T.ToTensor()(gray)
    dct = torch_dct.dct_2d(gray_t)
    mag = torch.log1p(torch.abs(dct)).squeeze(0).numpy()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mag, cmap="magma")
    ax.set_title("Log-Magnitude DCT Spectrum")
    ax.set_xlabel("Horizontal Frequency (Low -> High)")
    ax.set_ylabel("Vertical Frequency (Low -> High)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.caption(
        "Top-left corner is DC / low-frequency energy (overall lighting and smooth contours). "
        "Bottom-right represents high-frequency microtextures. AI generators often leave subtle "
        "checkerboard or periodic spikes in high-frequency bands."
    )

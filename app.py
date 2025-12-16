import streamlit as st
import torch
from PIL import Image
import numpy as np
import pathlib
from pathlib import Path

# --- OBAT KUAT WINDOWS PATH (WAJIB ADA DI PALING ATAS) ---
# Kode ini memaksa server Linux untuk bisa membaca path Windows dari file .pt kamu
pathlib.PosixPath = pathlib.WindowsPath
# ---------------------------------------------------------

st.set_page_config(page_title="Deteksi Uang", page_icon="💸")

# Judul
st.title("💸 Deteksi Keaslian Uang")
st.write("Mode: Local Source & Windows Path Fix")

# Sidebar
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Fungsi Load Model (GANTI JADI LOCAL)
@st.cache_resource
def load_model():
    try:
        # PENTING: source='local' artinya dia baca folder 'yolov5' yang kamu upload
        # Bukan download dari internet lagi.
        model = torch.hub.load(
            './yolov5', 
            'custom', 
            path='best_windows1.pt', 
            source='local'
        )
        return model
    except Exception as e:
        st.error(f"Error Load Model: {e}")
        return None

# Load Model
model = load_model()

# Input Gambar
input_image = None
pilihan = st.radio("Input:", ("Upload", "Kamera"))

if pilihan == "Upload":
    f = st.file_uploader("Upload gambar", type=['jpg','png','jpeg'])
    if f: input_image = Image.open(f)
else:
    c = st.camera_input("Foto")
    if c: input_image = Image.open(c)

# Deteksi
if input_image and model:
    if st.button('Deteksi'):
        model.conf = confidence
        results = model(input_image)
        st.image(results.render()[0], caption='Hasil', use_column_width=True)
        st.dataframe(results.pandas().xyxy[0])
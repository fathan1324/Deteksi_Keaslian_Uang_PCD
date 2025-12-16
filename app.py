import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Keaslian Uang",
    page_icon="💵",
    layout="wide"
)

# Fungsi untuk load model
@st.cache_resource
def load_model():
    try:
        # Load model YOLOv5 dari folder yolov5
        model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local', force_reload=True)
        model.conf = 0.5  # Confidence threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Fungsi untuk deteksi
def detect_money(image, model):
    # Konversi PIL Image ke array numpy
    img_array = np.array(image)
    
    # Deteksi dengan YOLOv5
    results = model(img_array)
    
    # Render hasil deteksi
    rendered_img = results.render()[0]
    
    # Konversi BGR ke RGB (OpenCV ke PIL)
    rendered_img_rgb = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
    
    # Get deteksi info
    detections = results.pandas().xyxy[0]
    
    return rendered_img_rgb, detections

# Header aplikasi
st.title("💵 Sistem Deteksi Keaslian Uang")
st.markdown("**Aplikasi berbasis YOLOv5 untuk mendeteksi keaslian uang kertas**")
st.markdown("---")

# Load model
with st.spinner("Loading model YOLOv5..."):
    model = load_model()

if model is None:
    st.error("⚠️ Gagal memuat model. Pastikan file `best.pt` dan folder `yolov5` ada di direktori yang benar.")
    st.stop()

st.success("✅ Model berhasil dimuat!")

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
model.conf = confidence

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Cara Penggunaan")
st.sidebar.markdown("""
1. Upload gambar uang kertas
2. Tunggu proses deteksi
3. Lihat hasil deteksi
4. Cek tingkat kepercayaan (confidence)
""")

# Main content
tab1, tab2 = st.tabs(["📷 Upload Gambar", "📹 Ambil Foto"])

with tab1:
    st.subheader("Upload Gambar Uang")
    uploaded_file = st.file_uploader(
        "Pilih file gambar...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar uang kertas yang ingin dideteksi"
    )
    
    if uploaded_file is not None:
        # Tampilkan gambar original
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📸 Gambar Original")
            st.image(image, use_container_width=True)
        
        # Tombol deteksi
        if st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True):
            with st.spinner("Sedang mendeteksi..."):
                # Proses deteksi
                result_img, detections = detect_money(image, model)
                
                with col2:
                    st.markdown("#### ✅ Hasil Deteksi")
                    st.image(result_img, use_container_width=True)
                
                # Tampilkan informasi deteksi
                st.markdown("---")
                st.subheader("📊 Detail Deteksi")
                
                if len(detections) > 0:
                    st.success(f"🎯 Terdeteksi {len(detections)} objek")
                    
                    # Tampilkan tabel hasil
                    for idx, det in detections.iterrows():
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 2, 1])
                            
                            with col_a:
                                st.metric("Kelas", det['name'])
                            with col_b:
                                confidence_pct = det['confidence'] * 100
                                st.metric("Confidence", f"{confidence_pct:.2f}%")
                            with col_c:
                                if confidence_pct > 70:
                                    st.success("✅ Tinggi")
                                elif confidence_pct > 50:
                                    st.warning("⚠️ Sedang")
                                else:
                                    st.error("❌ Rendah")
                    
                    # Tabel detail
                    st.dataframe(
                        detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']],
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ Tidak ada objek yang terdeteksi. Coba ubah confidence threshold atau gunakan gambar yang lebih jelas.")

with tab2:
    st.subheader("Ambil Foto dari Kamera")
    
    camera_input = st.camera_input("Ambil foto uang kertas")
    
    if camera_input is not None:
        # Tampilkan gambar dari kamera
        image = Image.open(camera_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📸 Foto dari Kamera")
            st.image(image, use_container_width=True)
        
        # Tombol deteksi
        if st.button("🔍 Deteksi Foto Ini", type="primary", use_container_width=True, key="detect_camera"):
            with st.spinner("Sedang mendeteksi..."):
                # Proses deteksi
                result_img, detections = detect_money(image, model)
                
                with col2:
                    st.markdown("#### ✅ Hasil Deteksi")
                    st.image(result_img, use_container_width=True)
                
                # Tampilkan informasi deteksi
                st.markdown("---")
                st.subheader("📊 Detail Deteksi")
                
                if len(detections) > 0:
                    st.success(f"🎯 Terdeteksi {len(detections)} objek")
                    
                    # Tampilkan hasil
                    for idx, det in detections.iterrows():
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 2, 1])
                            
                            with col_a:
                                st.metric("Kelas", det['name'])
                            with col_b:
                                confidence_pct = det['confidence'] * 100
                                st.metric("Confidence", f"{confidence_pct:.2f}%")
                            with col_c:
                                if confidence_pct > 70:
                                    st.success("✅ Tinggi")
                                elif confidence_pct > 50:
                                    st.warning("⚠️ Sedang")
                                else:
                                    st.error("❌ Rendah")
                    
                    st.dataframe(
                        detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']],
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ Tidak ada objek yang terdeteksi.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Dibuat dengan ❤️ menggunakan YOLOv5 dan Streamlit</p>
    <p><small>💡 Tip: Pastikan pencahayaan cukup dan uang terlihat jelas untuk hasil deteksi optimal</small></p>
</div>
""", unsafe_allow_html=True)

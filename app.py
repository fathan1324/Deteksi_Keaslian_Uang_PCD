import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import sys
import subprocess

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Keaslian Uang",
    page_icon="💵",
    layout="wide"
)

# Install ultralytics jika belum ada
@st.cache_resource
def install_dependencies():
    try:
        import ultralytics
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

# Fungsi untuk load model
@st.cache_resource
def load_model():
    try:
        # Install dependencies
        install_dependencies()
        
        from ultralytics import YOLO
        
        # Load model langsung dari best.pt
        model = YOLO('best.pt')
        
        st.success("✅ Model YOLOv5 berhasil dimuat!")
        return model
    except FileNotFoundError:
        st.error("❌ File 'best.pt' tidak ditemukan. Pastikan file ada di root directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Pastikan file 'best.pt' sudah di-upload ke repository GitHub")
        return None

# Fungsi untuk draw bounding boxes
def draw_boxes(image, results):
    """Draw bounding boxes on image using PIL"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    # Get boxes from results
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return img, []
    
    detections = []
    
    for idx, box in enumerate(boxes):
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get confidence and class
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        
        # Store detection info
        detections.append({
            'name': class_name,
            'confidence': confidence,
            'xmin': x1,
            'ymin': y1,
            'xmax': x2,
            'ymax': y2
        })
        
        # Choose color
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        text = f"{class_name} {confidence:.2f}"
        
        # Use default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Get text size
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background for text
        draw.rectangle([x1, y1-text_height-5, x1+text_width+5, y1], fill=color)
        
        # Draw text
        draw.text((x1+2, y1-text_height-3), text, fill='white', font=font)
    
    return img, detections

# Fungsi untuk deteksi
def detect_money(image, model, confidence_threshold):
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run detection
    results = model.predict(img_array, conf=confidence_threshold, verbose=False)
    
    # Draw boxes
    result_img, detections = draw_boxes(image, results)
    
    return result_img, detections

# Header aplikasi
st.title("💵 Sistem Deteksi Keaslian Uang")
st.markdown("**Aplikasi berbasis YOLOv5 untuk mendeteksi keaslian uang kertas**")
st.markdown("---")

# Load model
with st.spinner("⏳ Loading model YOLOv5... (first time might take a while)"):
    model = load_model()

if model is None:
    st.stop()

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Cara Penggunaan")
st.sidebar.markdown("""
1. Upload gambar uang kertas
2. Klik tombol "Deteksi Sekarang"
3. Lihat hasil deteksi
4. Cek tingkat kepercayaan (confidence)
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Pastikan pencahayaan cukup\n- Uang terlihat jelas\n- Tidak ada bayangan")

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
        # Load image
        image = Image.open(uploaded_file)
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📸 Gambar Original")
            st.image(image, use_container_width=True)
        
        # Tombol deteksi
        if st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True):
            with st.spinner("🔄 Sedang mendeteksi..."):
                try:
                    # Proses deteksi
                    result_img, detections = detect_money(image, model, confidence)
                    
                    with col2:
                        st.markdown("#### ✅ Hasil Deteksi")
                        st.image(result_img, use_container_width=True)
                    
                    # Tampilkan informasi deteksi
                    st.markdown("---")
                    st.subheader("📊 Detail Deteksi")
                    
                    if len(detections) > 0:
                        st.success(f"🎯 Terdeteksi **{len(detections)}** objek")
                        
                        # Tampilkan setiap deteksi
                        for idx, det in enumerate(detections):
                            with st.expander(f"🔍 Deteksi #{idx+1}: {det['name']}", expanded=True):
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
                                
                                # Koordinat
                                st.caption(f"📍 Koordinat: ({det['xmin']}, {det['ymin']}) → ({det['xmax']}, {det['ymax']})")
                        
                    else:
                        st.warning("⚠️ Tidak ada objek yang terdeteksi.")
                        st.info("💡 Coba:\n- Ubah confidence threshold di sidebar\n- Gunakan gambar yang lebih jelas\n- Pastikan pencahayaan cukup")
                
                except Exception as e:
                    st.error(f"❌ Error saat deteksi: {str(e)}")
                    st.info("Coba upload gambar lain atau refresh halaman")

with tab2:
    st.subheader("Ambil Foto dari Kamera")
    
    camera_input = st.camera_input("📸 Ambil foto uang kertas")
    
    if camera_input is not None:
        # Load image
        image = Image.open(camera_input)
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📸 Foto dari Kamera")
            st.image(image, use_container_width=True)
        
        # Tombol deteksi
        if st.button("🔍 Deteksi Foto Ini", type="primary", use_container_width=True, key="detect_camera"):
            with st.spinner("🔄 Sedang mendeteksi..."):
                try:
                    # Proses deteksi
                    result_img, detections = detect_money(image, model, confidence)
                    
                    with col2:
                        st.markdown("#### ✅ Hasil Deteksi")
                        st.image(result_img, use_container_width=True)
                    
                    # Tampilkan informasi deteksi
                    st.markdown("---")
                    st.subheader("📊 Detail Deteksi")
                    
                    if len(detections) > 0:
                        st.success(f"🎯 Terdeteksi **{len(detections)}** objek")
                        
                        # Tampilkan setiap deteksi
                        for idx, det in enumerate(detections):
                            with st.expander(f"🔍 Deteksi #{idx+1}: {det['name']}", expanded=True):
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
                                
                                st.caption(f"📍 Koordinat: ({det['xmin']}, {det['ymin']}) → ({det['xmax']}, {det['ymax']})")
                    else:
                        st.warning("⚠️ Tidak ada objek yang terdeteksi.")
                        st.info("💡 Coba ambil foto ulang dengan pencahayaan lebih baik")
                
                except Exception as e:
                    st.error(f"❌ Error saat deteksi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Dibuat dengan ❤️ menggunakan YOLOv5 (Ultralytics) dan Streamlit</p>
    <p><small>🔐 Data aman - Semua proses dilakukan di server</small></p>
</div>
""", unsafe_allow_html=True)

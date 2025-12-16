import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Keaslian Uang",
    page_icon="💵",
    layout="wide"
)

# Fungsi untuk load model YOLOv5
@st.cache_resource
def load_model():
    try:
        # Load YOLOv5 dari torch.hub (GitHub ultralytics/yolov5)
        # source='github' akan otomatis download repo YOLOv5
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.5  # Default confidence
        
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
def draw_boxes(image, detections):
    """Draw bounding boxes on image using PIL"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    if len(detections) == 0:
        return img
    
    for idx, det in detections.iterrows():
        # Get coordinates
        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        confidence = det['confidence']
        class_name = det['name']
        
        # Choose color based on class
        color = colors[idx % len(colors)]
        
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
    
    return img

# Fungsi untuk deteksi
def detect_money(image, model):
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run detection
    results = model(img_array)
    
    # Get detections as pandas dataframe
    detections = results.pandas().xyxy[0]
    
    # Draw boxes
    result_img = draw_boxes(image, detections)
    
    return result_img, detections

# Header aplikasi
st.title("💵 Sistem Deteksi Keaslian Uang")
st.markdown("**Aplikasi berbasis YOLOv5 untuk mendeteksi keaslian uang kertas**")
st.markdown("---")

# Load model
with st.spinner("⏳ Loading model YOLOv5... (first time might take 1-2 minutes)"):
    model = load_model()

if model is None:
    st.stop()

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
model.conf = confidence

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
                    result_img, detections = detect_money(image, model)
                    
                    with col2:
                        st.markdown("#### ✅ Hasil Deteksi")
                        st.image(result_img, use_container_width=True)
                    
                    # Tampilkan informasi deteksi
                    st.markdown("---")
                    st.subheader("📊 Detail Deteksi")
                    
                    if len(detections) > 0:
                        st.success(f"🎯 Terdeteksi **{len(detections)}** objek")
                        
                        # Tampilkan setiap deteksi
                        for idx, det in detections.iterrows():
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
                                st.caption(f"📍 Koordinat: ({int(det['xmin'])}, {int(det['ymin'])}) → ({int(det['xmax'])}, {int(det['ymax'])})")
                        
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
                    result_img, detections = detect_money(image, model)
                    
                    with col2:
                        st.markdown("#### ✅ Hasil Deteksi")
                        st.image(result_img, use_container_width=True)
                    
                    # Tampilkan informasi deteksi
                    st.markdown("---")
                    st.subheader("📊 Detail Deteksi")
                    
                    if len(detections) > 0:
                        st.success(f"🎯 Terdeteksi **{len(detections)}** objek")
                        
                        # Tampilkan setiap deteksi
                        for idx, det in detections.iterrows():
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
                                
                                st.caption(f"📍 Koordinat: ({int(det['xmin'])}, {int(det['ymin'])}) → ({int(det['xmax'])}, {int(det['ymax'])})")
                    else:
                        st.warning("⚠️ Tidak ada objek yang terdeteksi.")
                        st.info("💡 Coba ambil foto ulang dengan pencahayaan lebih baik")
                
                except Exception as e:
                    st.error(f"❌ Error saat deteksi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Dibuat dengan ❤️ menggunakan YOLOv5 dan Streamlit</p>
    <p><small>🔐 Data aman - Semua proses dilakukan di server</small></p>
</div>
""", unsafe_allow_html=True)

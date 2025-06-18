# File: app/frontend/app.py
# -*- coding: utf-8 -*-

import streamlit as st
import sys
import os

# --- Thêm đường dẫn dự án vào sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
# --- Kết thúc thêm đường dẫn ---

from app.backend.utils.face_detection import start_detection, stop_detection
from app.backend.utils.form_labels import show_form
from app.backend.utils.add_face import start_add_face
from app.backend.utils.pca_recognition import train_recognizer, start_recognition

st.title("PCA Recognition Web APP")

st.sidebar.header("Menu")
face_detection_btn = st.sidebar.button("Face Detection")
face_recognition_btn = st.sidebar.button("Face Recognition")
add_new_face_btn = st.sidebar.button("Add new face")
train_recognizer_btn = st.sidebar.button("Train Recognize")

# Khởi tạo các biến trạng thái cần thiết
if "page" not in st.session_state:
    st.session_state.page = "home"
if "label_name" not in st.session_state:
    st.session_state.label_name = ""
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False
if "train_status" not in st.session_state:
    st.session_state.train_status = None
# Thêm trạng thái cho các tham số
if "num_components" not in st.session_state:
    st.session_state.num_components = 30
if "k" not in st.session_state:
    st.session_state.k = 3
if "threshold" not in st.session_state:
    st.session_state.threshold = None  # Mặc định là None để dùng effective_threshold

# Xử lý điều hướng sidebar
if face_detection_btn:
    st.session_state.page = "face_detection"
    st.session_state.run_detection = False
    stop_detection()
elif face_recognition_btn:
    st.session_state.page = "face_recognition"
    stop_detection()
elif add_new_face_btn:
    st.session_state.page = "add_new_face"
    st.session_state.label_name = ""  # Reset label_name khi bắt đầu thêm mới
    stop_detection()
elif train_recognizer_btn:
    st.session_state.page = "train_recognizer"
    stop_detection()

# Hiển thị nội dung theo từng trang
if st.session_state.page == "home":
    st.write("Chào mừng bạn đến với hệ thống nhận dạng khuôn mặt sử dụng PCA và Streamlit.")
    st.write("Chọn chức năng từ menu bên trái để bắt đầu.")

elif st.session_state.page == "face_detection":
    st.header("Face Detection")
    st.write("Face Detection Using Haar-cascades")
    detection_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start Detection")
    with col2:
        stop_btn = st.button("Stop Detection")
    if start_btn:
        st.session_state.run_detection = True
        st.rerun()
    if stop_btn:
        stop_detection()
        st.session_state.run_detection = False
        st.session_state.page = "home"
        st.rerun()
    if st.session_state.get('run_detection', False):
        start_detection(detection_placeholder)

elif st.session_state.page == "add_new_face":
    st.header("Add new face")
    form_placeholder = st.empty()
    show_form(form_placeholder)

elif st.session_state.page == "add_face_process":
    st.info(f"Bạn đã nhập tên: {st.session_state.label_name}. Bắt đầu thêm khuôn mặt.")
    add_placeholder = st.empty()
    start_add_face(add_placeholder)

elif st.session_state.page == "face_recognition":
    st.header("Face Recognition")
    st.write("Face Recognition Using PCA")
    recognition_placeholder = st.empty()
    start_recognition(recognition_placeholder)

elif st.session_state.page == "train_recognizer":
    st.header("Train Recognizer")
    train_placeholder = st.empty()

    # Thanh điều chỉnh các tham số với giá trị từ session_state
    st.session_state.num_components = st.slider("Số thành phần chính (num_components)", min_value=5, max_value=50, value=st.session_state.num_components, step=5)
    st.session_state.k = st.slider("Số láng giềng (k)", min_value=1, max_value=10, value=st.session_state.k, step=1)
    st.session_state.threshold = st.slider("Ngưỡng (threshold)", min_value=0, max_value=5000, value=1000 if st.session_state.threshold is not None else 1000, step=100)

    # Nút Train
    if st.button("Train"):
        with st.spinner("Đang huấn luyện mô hình..."):
            try:
                effective_threshold = st.session_state.threshold if st.session_state.threshold != 0 else None
                train_recognizer(st.session_state.num_components, st.session_state.k, effective_threshold)
                st.session_state.train_status = "success"
                train_placeholder.write("Mô hình PCA đã được huấn luyện thành công!")
                if os.path.exists("app/backend/models/pca_model.pkl"):
                    train_placeholder.write("File pca_model.pkl đã được tạo.")
                else:
                    train_placeholder.write("Lỗi: File pca_model.pkl không được tạo.")
            except Exception as e:
                st.session_state.train_status = "error"
                train_placeholder.write(f"Lỗi khi huấn luyện: {str(e)}")
                st.write("Chi tiết lỗi:", str(e))  # Hiển thị lỗi chi tiết

    # Không tự động quay về home, chỉ quay về khi train xong hoặc chuyển trang thủ công
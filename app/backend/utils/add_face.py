# import cv2
# import streamlit as st
# import numpy as np
# from PIL import Image
# import os

# def add_face(placeholder):
#     # Lấy tên label từ session_state
#     label_name = st.session_state.get("label_name", "unknown")
#     save_dir = f"data/faces/{label_name}"
#     os.makedirs(save_dir, exist_ok=True)

#     # Khởi tạo camera
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise Exception("Không thể mở camera!")

#     # Load Haar-cascade classifier
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
#     )
#     if face_cascade.empty():
#         raise Exception("Không tìm thấy file Haar-cascade!")

#     start_time = cv2.getTickCount()
#     frame_count = 0
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # FPS mặc định là 30 nếu không lấy được
#     duration = 10  # 10 giây

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Chuyển sang ảnh trắng đen để detect
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         # Xử lý từng khuôn mặt detected
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             # Cắt khuôn mặt
#             face_roi = frame[y:y+h, x:x+w]
#             # Scale về 92x112 và chuyển về ảnh trắng đen
#             face_resized = cv2.resize(face_roi, (92, 112))
#             face_rgb = face_resized  # cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

#             # Lưu ảnh sau mỗi frame (tùy chọn, có thể giảm tần suất)
#             if frame_count % 5 == 0:  # Lưu 1 frame trên 5 frame để giảm số lượng ảnh
#                 filename = f"{save_dir}/face_{frame_count}.jpg"
#                 cv2.imwrite(filename, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

#         # Hiển thị frame trên Streamlit
#         if frame is not None:
#             frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             placeholder.image(frame_pil, channels="RGB")

#         # Kiểm tra thời gian 10 giây
#         current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
#         if current_time >= duration:
#             break

#         frame_count += 1

#         # Dừng khi nhấn Stop trên Streamlit
#         if not st.runtime.exists():
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     st.session_state.page = None  # Quay lại Home sau khi hoàn tất

# def start_add_face(placeholder):
#     add_face(placeholder)

import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os
import time

def add_face(placeholder):
    label_name = st.session_state.get("label_name", "unknown")
    save_dir = f"data/faces/{label_name}"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Không thể mở camera!")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        st.error("Không tìm thấy file Haar-cascade!")
        return

    start_time = time.time()
    frame_count = 0
    saved_count = 0
    max_images = 20
    duration = 15  # giây

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Vẽ bounding box lên frame màu để hiển thị
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            if w > 0 and h > 0 and x >= 0 and y >= 0 and x+w <= gray.shape[1] and y+h <= gray.shape[0]:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                # Chỉ lưu nếu vùng cắt hợp lệ và resize thành công
                if face_roi.size > 0:
                    try:
                        face_resized = cv2.resize(face_roi, (92, 112))
                        if saved_count < max_images and frame_count % 15 == 0:
                            filename = f"{save_dir}/face_{saved_count}.jpg"
                            success = cv2.imwrite(filename, face_resized)
                            if success:
                                saved_count += 1
                            else:
                                print(f"Lưu ảnh {filename} thất bại!")
                    except Exception as e:
                        print(f"Lỗi resize hoặc lưu ảnh: {e}")

        # Hiển thị frame đã vẽ bounding box lên Streamlit
        frame_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        placeholder.image(frame_pil, channels="RGB")

        # Thoát nếu đủ thời gian hoặc đủ số ảnh
        if (time.time() - start_time) >= duration or saved_count >= max_images:
            break

        frame_count += 1

        # Dừng khi nhấn Stop trên Streamlit
        if not st.runtime.exists():
            break

    cap.release()
    cv2.destroyAllWindows()
    st.session_state.page = "home"
    st.rerun()

def start_add_face(placeholder):
    add_face(placeholder)
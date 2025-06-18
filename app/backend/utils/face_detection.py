# # # import cv2
# # # import numpy as np
# # # from PIL import Image
# # # import io

# # # def detect_face():
# # #     # Khởi tạo camera
# # #     cap = cv2.VideoCapture(0)
# # #     if not cap.isOpened():
# # #         raise Exception("Không thể mở camera!")

# # #     # Load Haar-cascade classifier
# # #     face_cascade = cv2.CascadeClassifier(
# # #         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# # #     )
# # #     if face_cascade.empty():
# # #         raise Exception("Không tìm thấy file Haar-cascade!")

# # #     frame_placeholder = None
# # #     while True:
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break

# # #         # Chuyển sang ảnh xám để detect
# # #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# # #         # Vẽ bounding box quanh khuôn mặt
# # #         for (x, y, w, h) in faces:
# # #             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # #         # Chuyển frame thành hình ảnh để hiển thị trên Streamlit
# # #         if frame is not None:
# # #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # #             frame_pil = Image.fromarray(frame_rgb)
# # #             frame_placeholder.image(frame_pil, channels="RGB")

# # #         # Thoát khi nhấn 'q'
# # #         if cv2.waitKey(1) & 0xFF == ord('q'):
# # #             break

# # #     cap.release()
# # #     cv2.destroyAllWindows()

# # # def start_detection(placeholder):
# # #     global frame_placeholder
# # #     frame_placeholder = placeholder
# # #     detect_face()

# # import cv2
# # import numpy as np
# # from PIL import Image
# # import io

# # def detect_face(placeholder):
# #     # Khởi tạo camera
# #     cap = cv2.VideoCapture(0)
# #     if not cap.isOpened():
# #         raise Exception("Không thể mở camera!")

# #     # Load Haar-cascade classifier
# #     face_cascade = cv2.CascadeClassifier(
# #         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# #     )
# #     if face_cascade.empty():
# #         raise Exception("Không tìm thấy file Haar-cascade!")

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         # Chuyển sang ảnh xám để detect
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# #         # Vẽ bounding box quanh khuôn mặt
# #         for (x, y, w, h) in faces:
# #             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# #         # Chuyển frame thành hình ảnh để hiển thị trên Streamlit
# #         if frame is not None:
# #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #             frame_pil = Image.fromarray(frame_rgb)
# #             placeholder.image(frame_pil, channels="RGB")

# #         # Thoát khi nhấn 'q'
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # def start_detection(placeholder):
# #     detect_face(placeholder)

import cv2
import numpy as np
from PIL import Image
import streamlit as st # Thêm import streamlit để sử dụng session_state

# Sử dụng st.session_state để lưu trữ camera instance
# Điều này giúp Streamlit không phải khởi tạo lại camera mỗi khi script chạy lại
if 'cap' not in st.session_state:
    st.session_state.cap = None

def start_detection(placeholder):
    if "cap" not in st.session_state:
        st.session_state.cap = None
    """
    Hàm này bắt đầu luồng video và phát hiện khuôn mặt.
    Nó sẽ chạy trong một vòng lặp được điều khiển bởi nút "Stop" trên Streamlit.
    """
    # Khởi tạo camera nếu chưa được khởi tạo
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("Không thể mở camera! Vui lòng kiểm tra webcam hoặc quyền truy cập.")
            return

    # Load Haar-cascade classifier
    # Kiểm tra xem face_cascade đã được load vào session_state chưa để tránh load lại nhiều lần
    if 'face_cascade' not in st.session_state or st.session_state.face_cascade.empty():
        st.session_state.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if st.session_state.face_cascade.empty():
            st.error("Không tìm thấy file Haar-cascade! Vui lòng kiểm tra đường dẫn.")
            st.session_state.cap.release() # Giải phóng camera nếu không load được cascade
            st.session_state.cap = None
            return

    # Biến để kiểm soát vòng lặp (ví dụ: từ nút Stop)
    # Chúng ta sẽ sử dụng st.session_state để giữ trạng thái này
    if 'run_detection' not in st.session_state:
        st.session_state.run_detection = False

    st.session_state.run_detection = True # Bắt đầu chạy detection

    # Vòng lặp chính để xử lý từng frame
    # Vòng lặp này sẽ bị ngắt khi st.session_state.run_detection trở thành False
    while st.session_state.run_detection:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("Không thể đọc frame từ camera. Dừng phát hiện.")
            break

        # Chuyển sang ảnh xám để detect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = st.session_state.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Vẽ bounding box quanh khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Chuyển frame thành hình ảnh để hiển thị trên Streamlit
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", use_container_width=True) # use_column_width để ảnh hiển thị đẹp hơn

        # Quan trọng: Không dùng cv2.waitKey(1) ở đây
        # Streamlit sẽ tự động refresh trang và gọi hàm này lặp lại
        # Tuy nhiên, để dừng vòng lặp, chúng ta sẽ cần một nút Stop trong app.py

    # Dọn dẹp sau khi vòng lặp dừng
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        st.session_state.cap.release()
        st.session_state.cap = None # Đặt lại cap về None
    cv2.destroyAllWindows() # Lệnh này thường không cần thiết trong Streamlit

def stop_detection():
    """Hàm này dừng luồng phát hiện."""
    if 'run_detection' in st.session_state:
        st.session_state.run_detection = False
    if 'cap' in st.session_state and st.session_state.cap is not None and st.session_state.cap.isOpened():
        st.session_state.cap.release()
        st.session_state.cap = None
    st.info("Phát hiện khuôn mặt đã dừng.")

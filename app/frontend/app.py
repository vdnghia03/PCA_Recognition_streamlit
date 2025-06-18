
# import streamlit as st
# import sys
# import os

# # --- Bắt đầu phần thêm đường dẫn dự án vào sys.path ---
# # Lấy đường dẫn tuyệt đối của thư mục chứa file app.py
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Tính toán lại project_root
# # Từ app.py (trong frontend), đi lên 1 cấp -> app/
# # Đi lên 1 cấp nữa -> Web_App/ (Đây là thư mục gốc của dự án)
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # Chỉ còn 2 lần '..'

# # Thêm đường dẫn gốc của dự án vào sys.path
# sys.path.insert(0, project_root)
# # --- Kết thúc phần thêm đường dẫn ---


# # --- ĐÂY LÀ DÒNG CẦN THAY ĐỔI QUAN TRỌNG NHẤT ---
# # Thay đổi đường dẫn import từ app.utils thành app.backend.utils

# from app.backend.utils.face_detection import start_detection
# from app.backend.utils.form_labels import show_form
# from app.backend.utils.add_face import start_add_face

# # Tiêu đề ứng dụng
# st.title("PCA Recognition Web APP")

# # Sidebar với các chức năng
# st.sidebar.header("Menu")
# face_detection = st.sidebar.button("Face Detection")
# face_recognition = st.sidebar.button("Face Recognition")
# add_new_face = st.sidebar.button("Add new face")
# train_recognizer = st.sidebar.button("Train Recognizer")

# # Khu vực hiển thị nội dung chính
# st.write("Chọn chức năng từ menu bên trái để bắt đầu.")

# # Logic điều hướng
# if "page" not in st.session_state:
#     st.session_state.page = None

# if face_detection:
#     st.session_state.page = "face_detection"
# elif face_recognition:
#     st.session_state.page = "face_recognition"
# elif add_new_face:
#     st.session_state.page = "add_new_face"
# elif train_recognizer:
#     st.session_state.page = "train_recognizer"

# # Hiển thị trang tương ứng
# if st.session_state.page == "face_detection":
#     st.header("Face Detection")
#     st.write("Face Detection Using Haar-cascades")
#     detection_placeholder = st.empty()
#     start_detection(detection_placeholder)
# elif st.session_state.page == "add_new_face":
#     st.header("Add new face")
#     form_placeholder = st.empty()
#     if "label_name" not in st.session_state:
#         show_form(form_placeholder)
#     elif st.session_state.page == "add_face_process":
#         add_placeholder = st.empty()
#         start_add_face(add_placeholder)
# elif st.session_state.page == "face_recognition":
#     st.header("Face Recognition")
#     st.write("Chức năng nhận diện khuôn mặt sẽ được hiển thị ở đây.")
# elif st.session_state.page == "train_recognizer":
#     st.header("Train Recognizer")
#     st.write("Chức năng train mô hình sẽ được hiển thị ở đây.")

# import streamlit as st
# import sys
# import os

# # --- Bắt đầu phần thêm đường dẫn dự án vào sys.path ---
# # Lấy đường dẫn tuyệt đối của thư mục chứa file app.py
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Tính toán lại project_root
# # Từ app.py (trong frontend), đi lên 1 cấp -> app/
# # Đi lên 1 cấp nữa -> Web_App/ (Đây là thư mục gốc của dự án)
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # Chỉ còn 2 lần '..'

# # Thêm đường dẫn gốc của dự án vào sys.path
# sys.path.insert(0, project_root)
# # --- Kết thúc phần thêm đường dẫn ---


# # Import các hàm từ các module
# from app.backend.utils.face_detection import start_detection, stop_detection # Import cả stop_detection
# from app.backend.utils.form_labels import show_form
# from app.backend.utils.add_face import start_add_face

# # Tiêu đề ứng dụng
# st.title("PCA Recognition Web APP")

# # Sidebar với các chức năng
# st.sidebar.header("Menu")
# face_detection_btn = st.sidebar.button("Face Detection")
# face_recognition_btn = st.sidebar.button("Face Recognition")
# add_new_face_btn = st.sidebar.button("Add new face")
# train_recognizer_btn = st.sidebar.button("Train Recognizer")

# # Khu vực hiển thị nội dung chính
# if "page" not in st.session_state:
#     st.session_state.page = "home" # Trang mặc định

# # Logic điều hướng
# if face_detection_btn:
#     st.session_state.page = "face_detection"
#     # Đảm bảo dừng các luồng khác nếu có
#     stop_detection()
# elif face_recognition_btn:
#     st.session_state.page = "face_recognition"
#     stop_detection()
# elif add_new_face_btn:
#     st.session_state.page = "add_new_face"
#     stop_detection()
# elif train_recognizer_btn:
#     st.session_state.page = "train_recognizer"
#     stop_detection()

# # Hiển thị trang tương ứng
# if st.session_state.page == "home":
#     st.write("Chào mừng bạn đến với hệ thống nhận dạng khuôn mặt sử dụng PCA và Streamlit.")
#     st.write("Chọn chức năng từ menu bên trái để bắt đầu.")
# elif st.session_state.page == "face_detection":
#     st.header("Face Detection")
#     st.write("Face Detection Using Haar-cascades")

#     # Tạo một placeholder để hiển thị video
#     detection_placeholder = st.empty()

#     # Nút để bắt đầu/dừng phát hiện
#     col1, col2 = st.columns(2)
#     with col1:
#         start_btn = st.button("Start Detection")
#     with col2:
#         stop_btn = st.button("Stop Detection")

#     if start_btn:
#         st.session_state.run_detection = True
#         st.rerun() # Kích hoạt rerun để bắt đầu vòng lặp detection
    
#     if stop_btn:
#         stop_detection() # Gọi hàm dừng
#         st.session_state.page = "home" # Chuyển về trang chủ sau khi dừng
#         st.rerun() # Kích hoạt rerun để dừng vòng lặp và cập nhật UI

#     # Chỉ chạy detection nếu trạng thái cho phép
#     if st.session_state.get('run_detection', False):
#         start_detection(detection_placeholder) # Gọi hàm này trong vòng lặp của Streamlit

# elif st.session_state.page == "add_new_face":
#     st.header("Add new face")
#     form_placeholder = st.empty()
#     if "label_name" not in st.session_state:
#         st.session_state.label_name = "" # Khởi tạo label_name
    
#     # Hiển thị form để nhập tên nếu chưa có
#     if st.session_state.label_name == "":
#         show_form(form_placeholder)
#     else:
#         st.info(f"Bạn đã nhập tên: {st.session_state.label_name}. Bắt đầu thêm khuôn mặt.")
#         add_placeholder = st.empty()
#         start_add_face(add_placeholder, st.session_state.label_name) # Truyền tên vào hàm start_add_face

# elif st.session_state.page == "face_recognition":
#     st.header("Face Recognition")
#     st.write("Chức năng nhận diện khuôn mặt sẽ được hiển thị ở đây.")
# elif st.session_state.page == "train_recognizer":
#     st.header("Train Recognizer")
#     st.write("Chức năng train mô hình sẽ được hiển thị ở đây.")

# import streamlit as st
# import sys
# import os

# # --- Thêm đường dẫn dự án vào sys.path ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# sys.path.insert(0, project_root)
# # --- Kết thúc thêm đường dẫn ---

# from app.backend.utils.face_detection import start_detection, stop_detection
# from app.backend.utils.form_labels import show_form
# from app.backend.utils.add_face import start_add_face

# st.title("PCA Recognition Web APP")

# st.sidebar.header("Menu")
# face_detection_btn = st.sidebar.button("Face Detection")
# face_recognition_btn = st.sidebar.button("Face Recognition")
# add_new_face_btn = st.sidebar.button("Add new face")
# train_recognizer_btn = st.sidebar.button("Train Recognizer")

# # Khởi tạo các biến trạng thái cần thiết
# if "page" not in st.session_state:
#     st.session_state.page = "home"
# if "label_name" not in st.session_state:
#     st.session_state.label_name = ""
# if "run_detection" not in st.session_state:
#     st.session_state.run_detection = False

# # Xử lý điều hướng sidebar
# if face_detection_btn:
#     st.session_state.page = "face_detection"
#     st.session_state.run_detection = False
#     stop_detection()
# elif face_recognition_btn:
#     st.session_state.page = "face_recognition"
#     stop_detection()
# elif add_new_face_btn:
#     st.session_state.page = "add_new_face"
#     st.session_state.label_name = ""
#     stop_detection()
# elif train_recognizer_btn:
#     st.session_state.page = "train_recognizer"
#     stop_detection()

# # Hiển thị nội dung theo từng trang
# if st.session_state.page == "home":
#     st.write("Chào mừng bạn đến với hệ thống nhận dạng khuôn mặt sử dụng PCA và Streamlit.")
#     st.write("Chọn chức năng từ menu bên trái để bắt đầu.")

# elif st.session_state.page == "face_detection":
#     st.header("Face Detection")
#     st.write("Face Detection Using Haar-cascades")
#     detection_placeholder = st.empty()
#     col1, col2 = st.columns(2)
#     with col1:
#         start_btn = st.button("Start Detection")
#     with col2:
#         stop_btn = st.button("Stop Detection")
#     if start_btn:
#         st.session_state.run_detection = True
#         st.rerun()
#     if stop_btn:
#         stop_detection()
#         st.session_state.run_detection = False
#         st.session_state.page = "home"
#         st.rerun()
#     if st.session_state.get('run_detection', False):
#         start_detection(detection_placeholder)

# elif st.session_state.page == "add_new_face":
#     st.header("Add new face")
#     form_placeholder = st.empty()
#     show_form(form_placeholder)

# elif st.session_state.page == "add_face_process":
#     st.info(f"Bạn đã nhập tên: {st.session_state.label_name}. Bắt đầu thêm khuôn mặt.")
#     add_placeholder = st.empty()
#     start_add_face(add_placeholder)

# elif st.session_state.page == "face_recognition":
#     st.header("Face Recognition")
#     st.write("Chức năng nhận diện khuôn mặt sẽ được hiển thị ở đây.")

# elif st.session_state.page == "train_recognizer":
#     st.header("Train Recognizer")
#     st.write("Chức năng train mô hình sẽ được hiển thị ở đây.")

# import streamlit as st
# import sys
# import os

# # --- Thêm đường dẫn dự án vào sys.path ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# sys.path.insert(0, project_root)
# # --- Kết thúc thêm đường dẫn ---

# from app.backend.utils.face_detection import start_detection, stop_detection
# from app.backend.utils.form_labels import show_form
# from app.backend.utils.add_face import start_add_face

# st.title("PCA Recognition Web APP")

# st.sidebar.header("Menu")
# face_detection_btn = st.sidebar.button("Face Detection")
# face_recognition_btn = st.sidebar.button("Face Recognition")
# add_new_face_btn = st.sidebar.button("Add new face")
# train_recognizer_btn = st.sidebar.button("Train Recognizer")

# # Khởi tạo các biến trạng thái cần thiết
# if "page" not in st.session_state:
#     st.session_state.page = "home"
# if "label_name" not in st.session_state:
#     st.session_state.label_name = ""
# if "run_detection" not in st.session_state:
#     st.session_state.run_detection = False

# # Xử lý điều hướng sidebar
# if face_detection_btn:
#     st.session_state.page = "face_detection"
#     st.session_state.run_detection = False
#     stop_detection()
# elif face_recognition_btn:
#     st.session_state.page = "face_recognition"
#     stop_detection()
# elif add_new_face_btn:
#     st.session_state.page = "add_new_face"
#     st.session_state.label_name = ""  # Reset label_name khi bắt đầu thêm mới
#     stop_detection()
# elif train_recognizer_btn:
#     st.session_state.page = "train_recognizer"
#     stop_detection()

# # Hiển thị nội dung theo từng trang
# if st.session_state.page == "home":
#     st.write("Chào mừng bạn đến với hệ thống nhận dạng khuôn mặt sử dụng PCA và Streamlit.")
#     st.write("Chọn chức năng từ menu bên trái để bắt đầu.")

# elif st.session_state.page == "face_detection":
#     st.header("Face Detection")
#     st.write("Face Detection Using Haar-cascades")
#     detection_placeholder = st.empty()
#     col1, col2 = st.columns(2)
#     with col1:
#         start_btn = st.button("Start Detection")
#     with col2:
#         stop_btn = st.button("Stop Detection")
#     if start_btn:
#         st.session_state.run_detection = True
#         st.rerun()
#     if stop_btn:
#         stop_detection()
#         st.session_state.run_detection = False
#         st.session_state.page = "home"
#         st.rerun()
#     if st.session_state.get('run_detection', False):
#         start_detection(detection_placeholder)

# elif st.session_state.page == "add_new_face":
#     st.header("Add new face")
#     form_placeholder = st.empty()
#     show_form(form_placeholder)

# elif st.session_state.page == "add_face_process":
#     st.info(f"Bạn đã nhập tên: {st.session_state.label_name}. Bắt đầu thêm khuôn mặt.")
#     add_placeholder = st.empty()
#     start_add_face(add_placeholder)

# elif st.session_state.page == "face_recognition":
#     st.header("Face Recognition")
#     st.write("Chức năng nhận diện khuôn mặt sẽ được hiển thị ở đây.")

# elif st.session_state.page == "train_recognizer":
#     st.header("Train Recognizer")
#     st.write("Chức năng train mô hình sẽ được hiển thị ở đây.")

# import streamlit as st
# import sys
# import os

# # --- Thêm đường dẫn dự án vào sys.path ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# sys.path.insert(0, project_root)
# # --- Kết thúc thêm đường dẫn ---

# from app.backend.utils.face_detection import start_detection, stop_detection
# from app.backend.utils.form_labels import show_form
# from app.backend.utils.add_face import start_add_face
# from app.backend.utils.pca_recognition import train_recognizer, start_recognition

# st.title("PCA Recognition Web APP")

# st.sidebar.header("Menu")
# face_detection_btn = st.sidebar.button("Face Detection")
# face_recognition_btn = st.sidebar.button("Face Recognition")
# add_new_face_btn = st.sidebar.button("Add new face")
# train_recognizer_btn = st.sidebar.button("Train Recognizer")

# # Khởi tạo các biến trạng thái cần thiết
# if "page" not in st.session_state:
#     st.session_state.page = "home"
# if "label_name" not in st.session_state:
#     st.session_state.label_name = ""
# if "run_detection" not in st.session_state:
#     st.session_state.run_detection = False

# # Xử lý điều hướng sidebar
# if face_detection_btn:
#     st.session_state.page = "face_detection"
#     st.session_state.run_detection = False
#     stop_detection()
# elif face_recognition_btn:
#     st.session_state.page = "face_recognition"
#     stop_detection()
# elif add_new_face_btn:
#     st.session_state.page = "add_new_face"
#     st.session_state.label_name = ""  # Reset label_name khi bắt đầu thêm mới
#     stop_detection()
# elif train_recognizer_btn:
#     st.session_state.page = "train_recognizer"
#     stop_detection()

# # Hiển thị nội dung theo từng trang
# if st.session_state.page == "home":
#     st.write("Chào mừng bạn đến với hệ thống nhận dạng khuôn mặt sử dụng PCA và Streamlit.")
#     st.write("Chọn chức năng từ menu bên trái để bắt đầu.")

# elif st.session_state.page == "face_detection":
#     st.header("Face Detection")
#     st.write("Face Detection Using Haar-cascades")
#     detection_placeholder = st.empty()
#     col1, col2 = st.columns(2)
#     with col1:
#         start_btn = st.button("Start Detection")
#     with col2:
#         stop_btn = st.button("Stop Detection")
#     if start_btn:
#         st.session_state.run_detection = True
#         st.rerun()
#     if stop_btn:
#         stop_detection()
#         st.session_state.run_detection = False
#         st.session_state.page = "home"
#         st.rerun()
#     if st.session_state.get('run_detection', False):
#         start_detection(detection_placeholder)

# elif st.session_state.page == "add_new_face":
#     st.header("Add new face")
#     form_placeholder = st.empty()
#     show_form(form_placeholder)

# elif st.session_state.page == "add_face_process":
#     st.info(f"Bạn đã nhập tên: {st.session_state.label_name}. Bắt đầu thêm khuôn mặt.")
#     add_placeholder = st.empty()
#     start_add_face(add_placeholder)

# elif st.session_state.page == "face_recognition":
#     st.header("Face Recognition")
#     st.write("Face Recognition Using PCA")
#     recognition_placeholder = st.empty()
#     start_recognition(recognition_placeholder)

# elif st.session_state.page == "train_recognizer":
#     st.header("Train Recognizer")
#     train_placeholder = st.empty()
#     train_recognizer()
#     train_placeholder.write("Mô hình PCA đã được huấn luyện!")
#     st.session_state.page = "home"  # Quay lại Home sau khi train
#     st.rerun()

# import streamlit as st
# import sys
# import os

# # --- Thêm đường dẫn dự án vào sys.path ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# sys.path.insert(0, project_root)
# # --- Kết thúc thêm đường dẫn ---

# from app.backend.utils.face_detection import start_detection, stop_detection
# from app.backend.utils.form_labels import show_form
# from app.backend.utils.add_face import start_add_face
# from app.backend.utils.pca_recognition import train_recognizer, start_recognition

# st.title("PCA Recognition Web APP")

# st.sidebar.header("Menu")
# face_detection_btn = st.sidebar.button("Face Detection")
# face_recognition_btn = st.sidebar.button("Face Recognition")
# add_new_face_btn = st.sidebar.button("Add new face")
# train_recognizer_btn = st.sidebar.button("Train Recognizer")

# # Khởi tạo các biến trạng thái cần thiết
# if "page" not in st.session_state:
#     st.session_state.page = "home"
# if "label_name" not in st.session_state:
#     st.session_state.label_name = ""
# if "run_detection" not in st.session_state:
#     st.session_state.run_detection = False

# # Xử lý điều hướng sidebar
# if face_detection_btn:
#     st.session_state.page = "face_detection"
#     st.session_state.run_detection = False
#     stop_detection()
# elif face_recognition_btn:
#     st.session_state.page = "face_recognition"
#     stop_detection()
# elif add_new_face_btn:
#     st.session_state.page = "add_new_face"
#     st.session_state.label_name = ""  # Reset label_name khi bắt đầu thêm mới
#     stop_detection()
# elif train_recognizer_btn:
#     st.session_state.page = "train_recognizer"
#     stop_detection()

# # Hiển thị nội dung theo từng trang
# if st.session_state.page == "home":
#     st.write("Chào mừng bạn đến với hệ thống nhận dạng khuôn mặt sử dụng PCA và Streamlit.")
#     st.write("Chọn chức năng từ menu bên trái để bắt đầu.")

# elif st.session_state.page == "face_detection":
#     st.header("Face Detection")
#     st.write("Face Detection Using Haar-cascades")
#     detection_placeholder = st.empty()
#     col1, col2 = st.columns(2)
#     with col1:
#         start_btn = st.button("Start Detection")
#     with col2:
#         stop_btn = st.button("Stop Detection")
#     if start_btn:
#         st.session_state.run_detection = True
#         st.rerun()
#     if stop_btn:
#         stop_detection()
#         st.session_state.run_detection = False
#         st.session_state.page = "home"
#         st.rerun()
#     if st.session_state.get('run_detection', False):
#         start_detection(detection_placeholder)

# elif st.session_state.page == "add_new_face":
#     st.header("Add new face")
#     form_placeholder = st.empty()
#     show_form(form_placeholder)

# elif st.session_state.page == "add_face_process":
#     st.info(f"Bạn đã nhập tên: {st.session_state.label_name}. Bắt đầu thêm khuôn mặt.")
#     add_placeholder = st.empty()
#     start_add_face(add_placeholder)

# elif st.session_state.page == "face_recognition":
#     st.header("Face Recognition")
#     st.write("Face Recognition Using PCA")
#     recognition_placeholder = st.empty()
#     start_recognition(recognition_placeholder)

# elif st.session_state.page == "train_recognizer":
#     st.header("Train Recognizer")
#     train_placeholder = st.empty()
#     try:
#         train_recognizer()
#         train_placeholder.write("Mô hình PCA đã được huấn luyện thành công!")
#     except Exception as e:
#         train_placeholder.write(f"Lỗi khi huấn luyện: {e}")
#     st.session_state.page = "home"  # Quay lại Home sau khi train
#     st.rerun()

# import streamlit as st
# import sys
# import os

# # --- Thêm đường dẫn dự án vào sys.path ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# sys.path.insert(0, project_root)
# # --- Kết thúc thêm đường dẫn ---

# from app.backend.utils.face_detection import start_detection, stop_detection
# from app.backend.utils.form_labels import show_form
# from app.backend.utils.add_face import start_add_face
# from app.backend.utils.pca_recognition import train_recognizer, start_recognition

# st.title("PCA Recognition Web APP")

# st.sidebar.header("Menu")
# face_detection_btn = st.sidebar.button("Face Detection")
# face_recognition_btn = st.sidebar.button("Face Recognition")
# add_new_face_btn = st.sidebar.button("Add new face")
# train_recognizer_btn = st.sidebar.button("Train Recognizer")

# # Khởi tạo các biến trạng thái cần thiết
# if "page" not in st.session_state:
#     st.session_state.page = "home"
# if "label_name" not in st.session_state:
#     st.session_state.label_name = ""
# if "run_detection" not in st.session_state:
#     st.session_state.run_detection = False

# # Xử lý điều hướng sidebar
# if face_detection_btn:
#     st.session_state.page = "face_detection"
#     st.session_state.run_detection = False
#     stop_detection()
# elif face_recognition_btn:
#     st.session_state.page = "face_recognition"
#     stop_detection()
# elif add_new_face_btn:
#     st.session_state.page = "add_new_face"
#     st.session_state.label_name = ""  # Reset label_name khi bắt đầu thêm mới
#     stop_detection()
# elif train_recognizer_btn:
#     st.session_state.page = "train_recognizer"
#     stop_detection()

# # Hiển thị nội dung theo từng trang
# if st.session_state.page == "home":
#     st.write("Chào mừng bạn đến với hệ thống nhận dạng khuôn mặt sử dụng PCA và Streamlit.")
#     st.write("Chọn chức năng từ menu bên trái để bắt đầu.")

# elif st.session_state.page == "face_detection":
#     st.header("Face Detection")
#     st.write("Face Detection Using Haar-cascades")
#     detection_placeholder = st.empty()
#     col1, col2 = st.columns(2)
#     with col1:
#         start_btn = st.button("Start Detection")
#     with col2:
#         stop_btn = st.button("Stop Detection")
#     if start_btn:
#         st.session_state.run_detection = True
#         st.rerun()
#     if stop_btn:
#         stop_detection()
#         st.session_state.run_detection = False
#         st.session_state.page = "home"
#         st.rerun()
#     if st.session_state.get('run_detection', False):
#         start_detection(detection_placeholder)

# elif st.session_state.page == "add_new_face":
#     st.header("Add new face")
#     form_placeholder = st.empty()
#     show_form(form_placeholder)

# elif st.session_state.page == "add_face_process":
#     st.info(f"Bạn đã nhập tên: {st.session_state.label_name}. Bắt đầu thêm khuôn mặt.")
#     add_placeholder = st.empty()
#     start_add_face(add_placeholder)

# elif st.session_state.page == "face_recognition":
#     st.header("Face Recognition")
#     st.write("Face Recognition Using PCA")
#     recognition_placeholder = st.empty()
#     start_recognition(recognition_placeholder)

# elif st.session_state.page == "train_recognizer":
#     st.header("Train Recognizer")
#     train_placeholder = st.empty()
#     try:
#         train_recognizer()
#         train_placeholder.write("Mô hình PCA đã được huấn luyện thành công!")
#     except Exception as e:
#         train_placeholder.write(f"Lỗi khi huấn luyện: {e}")
#     st.session_state.page = "home"  # Quay lại Home sau khi train
#     st.rerun()

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
train_recognizer_btn = st.sidebar.button("Train Recognizer")

# Khởi tạo các biến trạng thái cần thiết
if "page" not in st.session_state:
    st.session_state.page = "home"
if "label_name" not in st.session_state:
    st.session_state.label_name = ""
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False
if "train_status" not in st.session_state:
    st.session_state.train_status = None  # Thêm trạng thái train

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
    try:
        with st.spinner("Đang huấn luyện mô hình..."):
            train_recognizer()
        st.session_state.train_status = "success"
        train_placeholder.write("Mô hình PCA đã được huấn luyện thành công!")
        # Kiểm tra file đã lưu
        if os.path.exists("app/backend/models/pca_model.pkl"):
            train_placeholder.write("File pca_model.pkl đã được tạo.")
        else:
            train_placeholder.write("Lỗi: File pca_model.pkl không được tạo.")
    except Exception as e:
        st.session_state.train_status = "error"
        train_placeholder.write(f"Lỗi khi huấn luyện: {str(e)}")
        st.write("Chi tiết lỗi:", str(e))  # Hiển thị lỗi chi tiết
    st.session_state.page = "home"  # Quay lại Home sau khi train
    st.rerun()
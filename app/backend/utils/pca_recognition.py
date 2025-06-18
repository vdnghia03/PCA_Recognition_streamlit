# import os
# import cv2
# import numpy as np
# import joblib
# from PIL import Image
# import streamlit as st
# from .knn_model import KNNModel

# class PCARecognizer:
#     def __init__(self, image_size=(92, 112), num_components=70):
#         self.image_size = image_size
#         self.num_components = num_components
#         self.N = image_size[0] * image_size[1]
#         self.mean_matrix = None
#         self.eigenfaces = None
#         self.model_path = "app/backend/models/pca_model.pkl"
#         self.eigen_path = "app/backend/models/eigenvectors.npy"
#         self.labels_path = "app/backend/models/labels.npy"
#         self.projected_path = "app/backend/models/projected_data.npy"
#         self.data_path = "data/faces/"
#         self.knn_model = None

#     def load_images(self):
#         images = []
#         labels = []
#         print(f"Scanning directory: {self.data_path}")
#         for label in os.listdir(self.data_path):
#             label_dir = os.path.join(self.data_path, label)
#             if os.path.isdir(label_dir):
#                 print(f"Found label directory: {label_dir}")
#                 for filename in os.listdir(label_dir):
#                     if filename.endswith(".jpg"):
#                         img_path = os.path.join(label_dir, filename)
#                         print(f"Loading image: {img_path}")
#                         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                         if img is not None:
#                             img = cv2.resize(img, self.image_size)
#                             images.append(img.flatten())
#                             labels.append(label)
#         if not images:
#             print("No images found in data/faces/")
#         return np.array(images).T, labels  # Transpose to match PCA input (N, M)

#     def train(self):
#         # Load và chuẩn bị dữ liệu
#         training_set, labels = self.load_images()
#         if len(training_set.T) == 0:
#             raise Exception("Không có dữ liệu để train!")

#         print(f"Training set shape: {training_set.shape}, Number of labels: {len(labels)}")
#         # Tính mean matrix
#         mean_matrix = np.mean(training_set, axis=1).reshape(-1, 1)
#         Phi = training_set - mean_matrix

#         # Tính eigenfaces
#         print("Calculating eigenfaces...")
#         A = Phi
#         ATA = A.T @ A
#         eigenvalues, eigenvectors = np.linalg.eigh(ATA)
#         idx = np.argsort(eigenvalues)[::-1]
#         eigenvalues = eigenvalues[idx]
#         eigenvectors = eigenvectors[:, idx]
#         non_zero_idx = eigenvalues > 1e-10
#         eigenvalues = eigenvalues[non_zero_idx]
#         eigenvectors = eigenvectors[:, non_zero_idx]

#         effective_samples = len(eigenvalues)
#         if self.num_components > effective_samples:
#             print(f"Adjusting num_components from {self.num_components} to {effective_samples} due to limited eigenvalues")
#             self.num_components = effective_samples

#         if len(eigenvalues) < self.num_components:
#             raise ValueError(f"Only {len(eigenvalues)} non-zero eigenvalues available, but {self.num_components} requested!")

#         self.eigenfaces = A @ eigenvectors[:, :self.num_components]
#         self.eigenfaces /= np.linalg.norm(self.eigenfaces, axis=0)
#         self.mean_matrix = mean_matrix
#         self.labels = np.array(labels)

#         # Tính projected data
#         print("Calculating projected data...")
#         projected_data = self.eigenfaces.T @ Phi

#         # Lưu dữ liệu
#         print(f"Saving to {self.model_path}, {self.eigen_path}, {self.labels_path}, {self.projected_path}")
#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#         np.save(self.eigen_path, self.eigenfaces)
#         np.save(self.labels_path, self.labels)
#         np.save(self.projected_path, projected_data)
#         joblib.dump(self, self.model_path)
#         print(f"Mô hình PCA đã được lưu tại {self.model_path}")

#     def load_model(self):
#         if os.path.exists(self.model_path):
#             try:
#                 loaded_recognizer = joblib.load(self.model_path)
#                 # Load dữ liệu đã lưu
#                 loaded_recognizer.eigenfaces = np.load(loaded_recognizer.eigen_path)
#                 loaded_recognizer.labels = np.load(loaded_recognizer.labels_path)
#                 loaded_recognizer.projected_data = np.load(loaded_recognizer.projected_path)
#                 # Khởi tạo KNN model
#                 if loaded_recognizer.projected_data is not None and loaded_recognizer.labels is not None:
#                     loaded_recognizer.knn_model = KNNModel(loaded_recognizer.projected_data, loaded_recognizer.labels)
#                 return loaded_recognizer
#             except Exception as e:
#                 print(f"Lỗi khi tải mô hình: {e}")
#                 return None
#         return None

#     def recognize(self, image):
#         if self.eigenfaces is None or self.knn_model is None:
#             raise Exception("Mô hình PCA hoặc KNN chưa được huấn luyện hoặc tải!")

#         # Chuẩn bị ảnh mới
#         img = cv2.resize(image, self.image_size)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_vector = img_gray.flatten().reshape(-1, 1)

#         # Trừ mean face và chiếu vào không gian PCA
#         phi_new = img_vector - self.mean_matrix
#         projected_vector = self.eigenfaces.T @ phi_new

#         # Nhận diện bằng KNN
#         label, distance = self.knn_model.recognize(projected_vector)
#         return label, distance  # Trả về cả label và distance

# def train_recognizer():
#     recognizer = PCARecognizer()
#     recognizer.train()

# def start_recognition(placeholder):
#     recognizer = PCARecognizer().load_model()
#     if recognizer is None:
#         placeholder.write("Mô hình PCA chưa được huấn luyện. Vui lòng train trước!")
#         return

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         placeholder.write("Không thể mở camera!")
#         return

#     # Load Haar-cascade classifier
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     if face_cascade.empty():
#         placeholder.write("Không tìm thấy file Haar-cascade!")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect khuôn mặt
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         # Xử lý từng khuôn mặt
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             face_roi = frame[y:y+h, x:x+w]
#             label, distance = recognizer.recognize(face_roi)  # Lấy cả label và distance
#             # Vẽ label và distance
#             text = f"{label} (Dist: {distance:.2f})"
#             cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # Hiển thị frame trên Streamlit
#         if frame is not None:
#             try:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame_pil = Image.fromarray(frame_rgb)
#                 placeholder.image(frame_pil, channels="RGB")
#             except Exception as e:
#                 placeholder.write(f"Lỗi hiển thị frame: {e}")

#         # Dừng khi nhấn Stop trên Streamlit
#         if not st.runtime.exists():
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# import os
# import cv2
# import numpy as np
# import joblib
# from PIL import Image
# import streamlit as st
# from .knn_model import KNNModel

# class PCARecognizer:
#     def __init__(self, image_size=(92, 112), num_components=30):
#         self.image_size = image_size
#         self.num_components = num_components
#         self.N = image_size[0] * image_size[1]
#         self.mean_matrix = None
#         self.eigenfaces = None
#         self.model_path = "app/backend/models/pca_model.pkl"
#         self.eigen_path = "app/backend/models/eigenvectors.npy"
#         self.labels_path = "app/backend/models/labels.npy"
#         self.projected_path = "app/backend/models/projected_data.npy"
#         self.data_path = "data/faces/"
#         self.knn_model = None
#         self.threshold = None  # Thêm thuộc tính để lưu threshold

#     def load_images(self):
#         images = []
#         labels = []
#         print(f"Scanning directory: {self.data_path}")
#         for label in os.listdir(self.data_path):
#             label_dir = os.path.join(self.data_path, label)
#             if os.path.isdir(label_dir):
#                 print(f"Found label directory: {label_dir}")
#                 for filename in os.listdir(label_dir):
#                     if filename.endswith(".jpg"):
#                         img_path = os.path.join(label_dir, filename)
#                         print(f"Loading image: {img_path}")
#                         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                         if img is not None:
#                             img = cv2.resize(img, self.image_size)
#                             images.append(img.flatten())
#                             labels.append(label)
#         if not images:
#             print("No images found in data/faces/")
#         return np.array(images).T, labels  # Transpose to match PCA input (N, M)

#     def train(self, k=3, threshold=None):
#         # Load và chuẩn bị dữ liệu
#         training_set, labels = self.load_images()
#         if len(training_set.T) == 0:
#             raise Exception("Không có dữ liệu để train!")

#         print(f"Training set shape: {training_set.shape}, Number of labels: {len(labels)}")
#         # Tính mean matrix
#         mean_matrix = np.mean(training_set, axis=1).reshape(-1, 1)
#         Phi = training_set - mean_matrix

#         # Tính eigenfaces
#         print("Calculating eigenfaces...")
#         A = Phi
#         ATA = A.T @ A
#         eigenvalues, eigenvectors = np.linalg.eigh(ATA)
#         idx = np.argsort(eigenvalues)[::-1]
#         eigenvalues = eigenvalues[idx]
#         eigenvectors = eigenvectors[:, idx]
#         non_zero_idx = eigenvalues > 1e-10
#         eigenvalues = eigenvalues[non_zero_idx]
#         eigenvectors = eigenvectors[:, non_zero_idx]

#         effective_samples = len(eigenvalues)
#         if self.num_components > effective_samples:
#             print(f"Adjusting num_components from {self.num_components} to {effective_samples} due to limited eigenvalues")
#             self.num_components = effective_samples

#         if len(eigenvalues) < self.num_components:
#             raise ValueError(f"Only {len(eigenvalues)} non-zero eigenvalues available, but {self.num_components} requested!")

#         self.eigenfaces = A @ eigenvectors[:, :self.num_components]
#         self.eigenfaces /= np.linalg.norm(self.eigenfaces, axis=0)
#         self.mean_matrix = mean_matrix
#         self.labels = np.array(labels)

#         # Tính projected data
#         print("Calculating projected data...")
#         projected_data = self.eigenfaces.T @ Phi

#         # Lưu dữ liệu
#         print(f"Saving to {self.model_path}, {self.eigen_path}, {self.labels_path}, {self.projected_path}")
#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#         np.save(self.eigen_path, self.eigenfaces)
#         np.save(self.labels_path, self.labels)
#         np.save(self.projected_path, projected_data)
#         self.threshold = threshold  # Lưu threshold vào instance
#         joblib.dump(self, self.model_path)
#         print(f"Mô hình PCA đã được lưu tại {self.model_path}")

#         # Khởi tạo KNN model với k và threshold
#         self.knn_model = KNNModel(projected_data, self.labels, k, threshold)

#     def load_model(self):
#         if os.path.exists(self.model_path):
#             try:
#                 print(f"Attempting to load model from {self.model_path}")
#                 loaded_recognizer = joblib.load(self.model_path)
#                 print(f"Model loaded, loading data from {self.eigen_path}, {self.labels_path}, {self.projected_path}")
#                 # Load dữ liệu đã lưu
#                 loaded_recognizer.eigenfaces = np.load(loaded_recognizer.eigen_path)
#                 loaded_recognizer.labels = np.load(loaded_recognizer.labels_path)
#                 loaded_recognizer.projected_data = np.load(loaded_recognizer.projected_path)
#                 # Khởi tạo KNN model với threshold đã lưu
#                 if loaded_recognizer.projected_data is not None and loaded_recognizer.labels is not None:
#                     loaded_recognizer.knn_model = KNNModel(loaded_recognizer.projected_data, loaded_recognizer.labels, k=3, threshold=loaded_recognizer.threshold)
#                 print("KNN model initialized successfully")
#                 return loaded_recognizer
#             except Exception as e:
#                 print(f"Lỗi khi tải mô hình: {e}")
#                 return None
#         else:
#             print(f"File {self.model_path} does not exist")
#             return None

#     def recognize(self, image):
#         if self.eigenfaces is None or self.knn_model is None:
#             raise Exception("Mô hình PCA hoặc KNN chưa được huấn luyện hoặc tải!")

#         # Chuẩn bị ảnh mới
#         img = cv2.resize(image, self.image_size)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_vector = img_gray.flatten().reshape(-1, 1)

#         # Trừ mean face và chiếu vào không gian PCA
#         phi_new = img_vector - self.mean_matrix
#         projected_vector = self.eigenfaces.T @ phi_new

#         # Nhận diện bằng KNN, truyền threshold hiện tại
#         label, distance = self.knn_model.recognize(projected_vector, self.knn_model.threshold)
#         return label, distance  # Trả về cả label và distance

# def train_recognizer(num_components, k, threshold):
#     recognizer = PCARecognizer(num_components=num_components)
#     recognizer.train(k, threshold)

# def start_recognition(placeholder):
#     recognizer = PCARecognizer().load_model()
#     if recognizer is None:
#         placeholder.write("Mô hình PCA chưa được huấn luyện. Vui lòng train trước!")
#         return

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         placeholder.write("Không thể mở camera!")
#         return

#     # Load Haar-cascade classifier
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     if face_cascade.empty():
#         placeholder.write("Không tìm thấy file Haar-cascade!")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect khuôn mặt
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         # Xử lý từng khuôn mặt
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             face_roi = frame[y:y+h, x:x+w]
#             label, distance = recognizer.recognize(face_roi)  # Lấy cả label và distance
#             # Vẽ label và distance
#             text = f"{label} (Dist: {distance:.2f})"
#             cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # Hiển thị frame trên Streamlit
#         if frame is not None:
#             try:
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame_pil = Image.fromarray(frame_rgb)
#                 placeholder.image(frame_pil, channels="RGB")
#             except Exception as e:
#                 placeholder.write(f"Lỗi hiển thị frame: {e}")

#         # Dừng khi nhấn Stop trên Streamlit
#         if not st.runtime.exists():
#             break

#     cap.release()
#     cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import joblib
from PIL import Image
import streamlit as st
from .knn_model import KNNModel

class PCARecognizer:
    def __init__(self, image_size=(92, 112), num_components=30):
        self.image_size = image_size
        self.num_components = num_components
        self.N = image_size[0] * image_size[1]
        self.mean_matrix = None
        self.eigenfaces = None
        self.model_path = "app/backend/models/pca_model.pkl"
        self.eigen_path = "app/backend/models/eigenvectors.npy"
        self.labels_path = "app/backend/models/labels.npy"
        self.projected_path = "app/backend/models/projected_data.npy"
        self.data_path = "data/faces/"
        self.knn_model = None
        self.threshold = None  # Thêm thuộc tính để lưu threshold

    def load_images(self):
        images = []
        labels = []
        print(f"Scanning directory: {self.data_path}")
        for label in os.listdir(self.data_path):
            label_dir = os.path.join(self.data_path, label)
            if os.path.isdir(label_dir):
                print(f"Found label directory: {label_dir}")
                for filename in os.listdir(label_dir):
                    if filename.endswith(".jpg"):
                        img_path = os.path.join(label_dir, filename)
                        print(f"Loading image: {img_path}")
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, self.image_size)
                            images.append(img.flatten())
                            labels.append(label)
        if not images:
            print("No images found in data/faces/")
        return np.array(images).T, labels  # Transpose to match PCA input (N, M)

    def train(self, k=3, threshold=None):
        # Load và chuẩn bị dữ liệu
        training_set, labels = self.load_images()
        if len(training_set.T) == 0:
            raise Exception("Không có dữ liệu để train!")

        print(f"Training set shape: {training_set.shape}, Number of labels: {len(labels)}")
        # Tính mean matrix
        mean_matrix = np.mean(training_set, axis=1).reshape(-1, 1)
        Phi = training_set - mean_matrix

        # Tính eigenfaces
        print("Calculating eigenfaces...")
        A = Phi
        ATA = A.T @ A
        eigenvalues, eigenvectors = np.linalg.eigh(ATA)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        non_zero_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[non_zero_idx]
        eigenvectors = eigenvectors[:, non_zero_idx]

        effective_samples = len(eigenvalues)
        if self.num_components > effective_samples:
            print(f"Adjusting num_components from {self.num_components} to {effective_samples} due to limited eigenvalues")
            self.num_components = effective_samples

        if len(eigenvalues) < self.num_components:
            raise ValueError(f"Only {len(eigenvalues)} non-zero eigenvalues available, but {self.num_components} requested!")

        self.eigenfaces = A @ eigenvectors[:, :self.num_components]
        self.eigenfaces /= np.linalg.norm(self.eigenfaces, axis=0)
        self.mean_matrix = mean_matrix
        self.labels = np.array(labels)

        # Tính projected data
        print("Calculating projected data...")
        projected_data = self.eigenfaces.T @ Phi

        # Lưu dữ liệu
        print(f"Saving to {self.model_path}, {self.eigen_path}, {self.labels_path}, {self.projected_path}")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        np.save(self.eigen_path, self.eigenfaces)
        np.save(self.labels_path, self.labels)
        np.save(self.projected_path, projected_data)
        self.threshold = threshold  # Lưu threshold vào instance
        joblib.dump(self, self.model_path)
        print(f"Mô hình PCA đã được lưu tại {self.model_path}")

        # Khởi tạo KNN model với k và threshold
        self.knn_model = KNNModel(projected_data, self.labels, k, threshold)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"Attempting to load model from {self.model_path}")
                loaded_recognizer = joblib.load(self.model_path)
                print(f"Model loaded, loading data from {self.eigen_path}, {self.labels_path}, {self.projected_path}")
                # Load dữ liệu đã lưu
                loaded_recognizer.eigenfaces = np.load(loaded_recognizer.eigen_path)
                loaded_recognizer.labels = np.load(loaded_recognizer.labels_path)
                loaded_recognizer.projected_data = np.load(loaded_recognizer.projected_path)
                # Khởi tạo KNN model với threshold đã lưu
                if loaded_recognizer.projected_data is not None and loaded_recognizer.labels is not None:
                    loaded_recognizer.knn_model = KNNModel(loaded_recognizer.projected_data, loaded_recognizer.labels, k=3, threshold=loaded_recognizer.threshold)
                print("KNN model initialized successfully")
                return loaded_recognizer
            except Exception as e:
                print(f"Lỗi khi tải mô hình: {e}")
                return None
        else:
            print(f"File {self.model_path} does not exist")
            return None

    def recognize(self, image):
        if self.eigenfaces is None or self.knn_model is None:
            raise Exception("Mô hình PCA hoặc KNN chưa được huấn luyện hoặc tải!")

        # Chuẩn bị ảnh mới
        img = cv2.resize(image, self.image_size)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_vector = img_gray.flatten().reshape(-1, 1)

        # Trừ mean face và chiếu vào không gian PCA
        phi_new = img_vector - self.mean_matrix
        projected_vector = self.eigenfaces.T @ phi_new

        # Nhận diện bằng KNN, truyền threshold hiện tại
        label, distance = self.knn_model.recognize(projected_vector, self.knn_model.threshold)
        return label, distance  # Trả về cả label và distance

def train_recognizer(num_components, k, threshold):
    recognizer = PCARecognizer(num_components=num_components)
    recognizer.train(k, threshold)

def start_recognition(placeholder):
    recognizer = PCARecognizer().load_model()
    if recognizer is None:
        placeholder.write("Mô hình PCA chưa được huấn luyện. Vui lòng train trước!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        placeholder.write("Không thể mở camera!")
        return

    # Load Haar-cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        placeholder.write("Không tìm thấy file Haar-cascade!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Xử lý từng khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            label, distance = recognizer.recognize(face_roi)  # Lấy cả label và distance
            # Vẽ label và distance
            text = f"{label} (Dist: {distance:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị frame trên Streamlit
        if frame is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                placeholder.image(frame_pil, channels="RGB")
            except Exception as e:
                placeholder.write(f"Lỗi hiển thị frame: {e}")

        # Dừng khi nhấn Stop trên Streamlit
        if not st.runtime.exists():
            break

    cap.release()
    cv2.destroyAllWindows()
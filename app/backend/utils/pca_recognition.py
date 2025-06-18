

# import os
# import cv2
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import joblib
# from PIL import Image

# class PCARecognizer:
#     def __init__(self, image_size=(92, 112), num_components=50):
#         self.image_size = image_size
#         self.num_components = num_components
#         self.N = image_size[0] * image_size[1]
#         self.pca = None
#         self.knn = None
#         self.model_path = "app/backend/models/pca_model.pkl"

#     def load_images(self, data_path="data/faces/"):
#         images = []
#         labels = []
#         for label in os.listdir(data_path):
#             label_dir = os.path.join(data_path, label)
#             if os.path.isdir(label_dir):
#                 for filename in os.listdir(label_dir):
#                     if filename.endswith(".jpg"):
#                         img_path = os.path.join(label_dir, filename)
#                         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                         if img is not None:
#                             img = cv2.resize(img, self.image_size)
#                             images.append(img.flatten())
#                             labels.append(label)
#         return np.array(images).T, labels  # Transpose to match your PCA input (N, M)

#     def train(self):
#         # Load và chuẩn bị dữ liệu
#         training_set, labels = self.load_images()
#         if len(training_set.T) == 0:
#             raise Exception("Không có dữ liệu để train!")

#         # Khởi tạo và huấn luyện PCA
#         self.pca = PCA(training_set, labels, self.num_components, self.image_size)
#         self.knn = self.pca._train_knn()

#         # Lưu mô hình
#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#         joblib.dump(self, self.model_path)
#         print(f"Mô hình PCA đã được lưu tại {self.model_path}")

#     def load_model(self):
#         if os.path.exists(self.model_path):
#             try:
#                 return joblib.load(self.model_path)
#             except Exception as e:
#                 print(f"Lỗi khi tải mô hình: {e}")
#                 return None
#         return None

#     def recognize(self, image, threshold=1000):
#         if self.pca is None or self.knn is None:
#             raise Exception("Mô hình PCA hoặc KNN chưa được huấn luyện hoặc tải!")

#         # Chuẩn bị ảnh mới
#         img = cv2.resize(image, self.image_size)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_vector = img_gray.flatten().reshape(-1, 1)

#         # Nhận diện
#         label = self.pca.recognize_face(img_vector, threshold=threshold)
#         return label

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
#             label = recognizer.recognize(face_roi)
#             cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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

# # Định nghĩa lớp PCA của bạn
# class PCA:
#     def __init__(self, training_set, labels, num_components, image_size=(92, 112)):
#         if num_components >= training_set.shape[1]:
#             raise ValueError("Number of components must be less than number of samples!")

#         self.training_set = training_set
#         self.labels = labels
#         self.num_components = num_components
#         self.image_size = image_size
#         self.N = image_size[0] * image_size[1]

#         self.mean_matrix = self._get_mean(training_set)
#         self.eigenfaces, self.eigenvalues = self._get_eigenfaces(training_set, num_components)
#         self.knn = self._train_knn()

#     def _get_mean(self, input_data):
#         """Step 3: Compute mean face vector (Psi)."""
#         mean = np.mean(input_data, axis=1).reshape(-1, 1)
#         return mean

#     def _get_eigenfaces(self, input_data, K):
#         """Steps 4-7: Compute eigenfaces (u_i) and eigenvalues."""
#         M = input_data.shape[1]
#         Phi = input_data - self.mean_matrix
#         A = Phi
#         ATA = A.T @ A
#         eigenvalues, eigenvectors = np.linalg.eigh(ATA)
#         idx = np.argsort(eigenvalues)[::-1]
#         eigenvalues = eigenvalues[idx]
#         eigenvectors = eigenvectors[:, idx]
#         non_zero_idx = eigenvalues > 1e-10
#         eigenvalues = eigenvalues[non_zero_idx]
#         eigenvectors = eigenvectors[:, non_zero_idx]

#         if len(eigenvalues) < K:
#             raise ValueError(f"Only {len(eigenvalues)} non-zero eigenvalues available, but K={K} requested!")

#         u = A @ eigenvectors[:, :K]
#         u /= np.linalg.norm(u, axis=0)
#         return u, eigenvalues[:K]

#     def get_eigenfaces(self):
#         """Return eigenfaces matrix (W)."""
#         return self.eigenfaces

#     def get_eigenvalues(self):
#         """Return eigenvalues."""
#         return self.eigenvalues

#     def get_mean_matrix(self):
#         """Return mean face vector."""
#         return self.mean_matrix

#     def get_projected_data(self):
#         """Return projected data (w_i = eigenfaces^T * Phi_i)."""
#         Phi = self.training_set - self.mean_matrix
#         return self.eigenfaces.T @ Phi

#     def _train_knn(self, k=3):
#         """Train KNN classifier on projected data."""
#         projected_data = self.get_projected_data()
#         X = projected_data.T
#         y = np.array(self.labels)
#         knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#         knn.fit(X, y)
#         return knn

#     def recognize_face(self, new_image_vector, k=3, threshold=None):
#         if new_image_vector.shape != (self.N, 1):
#             raise ValueError(f"New image has incorrect shape: {new_image_vector.shape}, expected ({self.N}, 1)")

#         phi_new = new_image_vector - self.mean_matrix
#         w_new = self.eigenfaces.T @ phi_new
#         X_new = w_new.flatten().reshape(1, -1)
#         predicted_label = self.knn.predict(X_new)[0]
#         distances, indices = self.knn.kneighbors(X_new, n_neighbors=1)
#         closest_distance = distances[0][0]

#         if threshold is not None and closest_distance > threshold:
#             return "unknown"
#         else:
#             return predicted_label

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
#         label, distance = self.knn_model.recognize(projected_vector,1000)
#         return label

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
#             label = recognizer.recognize(face_roi)
#             cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
    def __init__(self, image_size=(92, 112), num_components=50):
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

    def train(self):
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
        joblib.dump(self, self.model_path)
        print(f"Mô hình PCA đã được lưu tại {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                loaded_recognizer = joblib.load(self.model_path)
                # Load dữ liệu đã lưu
                loaded_recognizer.eigenfaces = np.load(loaded_recognizer.eigen_path)
                loaded_recognizer.labels = np.load(loaded_recognizer.labels_path)
                loaded_recognizer.projected_data = np.load(loaded_recognizer.projected_path)
                # Khởi tạo KNN model
                if loaded_recognizer.projected_data is not None and loaded_recognizer.labels is not None:
                    loaded_recognizer.knn_model = KNNModel(loaded_recognizer.projected_data, loaded_recognizer.labels)
                return loaded_recognizer
            except Exception as e:
                print(f"Lỗi khi tải mô hình: {e}")
                return None
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

        # Nhận diện bằng KNN
        label, distance = self.knn_model.recognize(projected_vector)
        return label, distance  # Trả về cả label và distance

def train_recognizer():
    recognizer = PCARecognizer()
    recognizer.train()

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
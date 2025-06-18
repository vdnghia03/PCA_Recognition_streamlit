# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# class KNNModel:
#     def __init__(self, projected_data, labels, k=3):
#         self.projected_data = projected_data
#         self.labels = labels
#         self.knn = self._train_knn(k)
#         self.threshold = self._calculate_threshold()  # Tính ngưỡng tự động

#     def _train_knn(self, k=3):
#         """Train KNN classifier on projected data."""
#         X = self.projected_data.T  # Chuyển về (M, K)
#         y = np.array(self.labels)
#         knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#         knn.fit(X, y)
#         return knn

#     def _calculate_threshold(self):
#         """Tính ngưỡng dựa trên khoảng cách trung bình giữa các mẫu."""
#         X = self.projected_data.T
#         distances = []
#         for i in range(len(X)):
#             for j in range(i + 1, len(X)):
#                 if self.labels[i] == self.labels[j]:  # Chỉ tính giữa các mẫu cùng label
#                     dist = np.linalg.norm(X[i] - X[j])
#                     distances.append(dist)
#         return np.mean(distances) * 1.5 if distances else 1000  # Ngưỡng = 1.5 lần trung bình

#     def recognize(self, projected_vector, threshold=None):
#         """Nhận diện dựa trên vector đã chiếu, trả về label hoặc 'unknown'."""
#         X_new = projected_vector.reshape(1, -1)
#         predicted_label = self.knn.predict(X_new)[0]
#         distances, indices = self.knn.kneighbors(X_new, n_neighbors=1)
#         closest_distance = distances[0][0]

#         threshold = self.threshold if threshold is None else threshold
#         if closest_distance > threshold:
#             return "unknown", closest_distance
#         return predicted_label, closest_distance

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self, projected_data, labels, k=3):
        self.projected_data = projected_data
        self.labels = labels
        self.knn = self._train_knn(k)
        self.threshold = self._calculate_threshold()  # Tính ngưỡng tự động

    def _train_knn(self, k=3):
        """Train KNN classifier on projected data."""
        X = self.projected_data.T  # Chuyển về (M, K)
        y = np.array(self.labels)
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X, y)
        return knn

    def _calculate_threshold(self):
        """Tính ngưỡng dựa trên phân vị 95th percentile của khoảng cách giữa các mẫu."""
        X = self.projected_data.T
        distances = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if self.labels[i] == self.labels[j]:  # Chỉ tính giữa các mẫu cùng label
                    dist = np.linalg.norm(X[i] - X[j])
                    distances.append(dist)
        if not distances:
            return 1000  # Giá trị mặc định nếu không có dữ liệu
        # Sử dụng percentile 95 để đặt ngưỡng chặt chẽ hơn
        threshold = np.percentile(distances, 95) * 1.2  # Nhân 1.2 để tăng độ an toàn
        return max(threshold, 500)  # Đảm bảo ngưỡng không quá thấp

    def recognize(self, projected_vector, threshold=None):
        """Nhận diện dựa trên vector đã chiếu, trả về label hoặc 'unknown'."""
        X_new = projected_vector.reshape(1, -1)
        predicted_label = self.knn.predict(X_new)[0]
        distances, indices = self.knn.kneighbors(X_new, n_neighbors=1)
        closest_distance = distances[0][0]

        threshold = self.threshold if threshold is None else threshold
        if closest_distance > threshold:
            return "unknown", closest_distance
        return predicted_label, closest_distance
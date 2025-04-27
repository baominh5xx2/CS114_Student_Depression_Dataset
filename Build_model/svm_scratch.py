import numpy as np

class LinearSVM_Dual:
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000):
        # Tham số C: hệ số phạt, tol: ngưỡng hội tụ, max_iter: số vòng lặp tối đa
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.astype(np.float64)
        # Chuyển nhãn về -1 và 1 (SVM yêu cầu)
        y = np.where(y == 0, -1, 1)

        # Tính ma trận Gram (kernel tuyến tính)
        K = X @ X.T

        # Khởi tạo các hệ số Lagrange alpha và bias b
        alpha = np.zeros(n_samples)
        b = 0

        for _ in range(self.max_iter):
            alpha_prev = np.copy(alpha)
            for i in range(n_samples):
                # Tính giá trị dự đoán cho mẫu i
                E_i = (alpha * y) @ K[:, i] + b - y[i]
                # Kiểm tra điều kiện KKT để cập nhật alpha
                if (y[i]*E_i < -self.tol and alpha[i] < self.C) or (y[i]*E_i > self.tol and alpha[i] > 0):
                    # Chọn ngẫu nhiên một chỉ số j khác i
                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    E_j = (alpha * y) @ K[:, j] + b - y[j]

                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    # Tính giới hạn L, H cho alpha[j]
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue

                    # Tính eta (hệ số cập nhật)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Cập nhật alpha[j]
                    alpha[j] -= y[j] * (E_i - E_j) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    # Cập nhật alpha[i]
                    alpha[i] += y[i]*y[j]*(alpha_j_old - alpha[j])

                    # Cập nhật bias b
                    b1 = b - E_i - y[i]*(alpha[i] - alpha_i_old)*K[i, i] - y[j]*(alpha[j] - alpha_j_old)*K[i, j]
                    b2 = b - E_j - y[i]*(alpha[i] - alpha_i_old)*K[i, j] - y[j]*(alpha[j] - alpha_j_old)*K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

            # Kiểm tra hội tụ
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.tol:
                break

        # Lưu lại các tham số mô hình
        self.alpha = alpha
        self.b = b
        self.X = X
        self.y = y
        # Tính vector trọng số w cho kernel tuyến tính
        self.w = ((alpha * y) @ X)

    def project(self, X):
        # Hàm dự đoán giá trị đầu ra (chưa lấy dấu)
        return X @ self.w + self.b

    def predict(self, X):
        # Hàm dự đoán nhãn (0 hoặc 1)
        return (self.project(X) >= 0).astype(int)


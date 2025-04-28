# 1. Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Hàm tính log loss
def compute_log_loss(y_true, y_pred):
    epsilon = 1e-15  # Tránh log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 3. Hàm gradient descent
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    # Khởi tạo weights kiểu float
    w = np.zeros(n, dtype=float)
    for i in range(epochs):
        z = X.dot(w)
        y_pred = sigmoid(z)
        grad = (X.T.dot(y_pred - y)) / m
        w -= lr * grad
    return w

# 4. Hàm predict sử dụng threshold
def predict(X, w, threshold=0.5):
    y_prob = sigmoid(X.dot(w))
    return (y_prob >= threshold).astype(int)

# 5. Chuẩn bị dữ liệu
# Lấy y và ép kiểu int
y_train_np = y_train.values.astype(int)
y_valid_np = y_valid.values.astype(int)
y_test_np = y_test.values.astype(int)

# Lấy X_raw và ép kiểu float
X_train_np = X_train_encoded.values.astype(float)
X_valid_np = X_valid_encoded.values.astype(float)
X_test_np = X_test_encoded.values.astype(float)

# Thêm cột bias (intercept) là 1, kiểu float
ones_train = np.ones((X_train_np.shape[0], 1), dtype=float)
ones_valid = np.ones((X_valid_np.shape[0], 1), dtype=float)
ones_test = np.ones((X_test_np.shape[0], 1), dtype=float)

X_train_np = np.hstack([ones_train, X_train_np])
X_valid_np = np.hstack([ones_valid, X_valid_np])
X_test_np = np.hstack([ones_test, X_test_np])

# 6. Huấn luyện
w = gradient_descent(X_train_np, y_train_np, lr=0.01, epochs=5000)

# 7. Dự đoán & đánh giá
y_train_pred = predict(X_train_np, w, threshold=0.5)
y_valid_pred = predict(X_valid_np, w, threshold=0.5)
y_test_pred = predict(X_test_np, w, threshold=0.5)

train_accuracy = np.mean(y_train_pred == y_train_np)
valid_accuracy = np.mean(y_valid_pred == y_valid_np)
test_accuracy = np.mean(y_test_pred == y_test_np)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
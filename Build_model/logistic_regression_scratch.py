import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, C=0.1, threshold=0.5):
        """
        Initialize Logistic Regression model with hyperparameters
        
        Parameters:
        - lr: learning rate for gradient descent
        - epochs: number of training iterations
        - C: inverse of regularization strength (like in sklearn)
            smaller values specify stronger regularization
        - threshold: decision threshold for binary classification
        """
        self.lr = lr
        self.epochs = epochs
        self.C = C  # Inverse regularization parameter
        self.threshold = threshold
        self.w = None
    
    def sigmoid(self, z):
        """Apply sigmoid function to input"""
        return 1 / (1 + np.exp(-z))
    
    def compute_log_loss(self, y_true, y_pred):
        """Calculate log loss (binary cross-entropy)"""
        epsilon = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        """
        Train logistic regression model using gradient descent
        
        Parameters:
        - X: feature matrix with bias column
        - y: target values (0 or 1)
        """
        # Add bias term if not present
        if np.all(X[:, 0] != 1):
            X = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
            
        m, n = X.shape
        # Initialize weights
        self.w = np.zeros(n, dtype=float)
        
        # Gradient descent
        for i in range(self.epochs):
            z = X.dot(self.w)
            y_pred = self.sigmoid(z)
            
            # Calculate gradient with L2 regularization (except for bias term)
            grad = (X.T.dot(y_pred - y)) / m
            
            # Add L2 regularization term (but not for bias term)
            if self.C > 0:
                reg_term = np.zeros(n)
                reg_term[1:] = self.w[1:] / self.C  # Don't regularize bias term
                grad += reg_term / m
                
            self.w -= self.lr * grad
            
        return self
    
    def predict_proba(self, X):
        """
        Predict probability of positive class
        
        Parameters:
        - X: feature matrix with bias column
        """
        # Add bias term if not present
        if np.all(X[:, 0] != 1):
            X = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
            
        return self.sigmoid(X.dot(self.w))
    
    def predict(self, X):
        """
        Predict class labels (0 or 1)
        
        Parameters:
        - X: feature matrix with bias column
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

def save_metrics_to_csv(model_name, y_true, y_pred, filepath='../model_metrics.csv'):
    """
    Save model metrics to CSV file with each model as a row
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    filepath : str
        Path to CSV file
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Class 0 metrics
    precision_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0)
    
    # Class 1 metrics
    precision_1 = precision_score(y_true, y_pred, pos_label=1)
    recall_1 = recall_score(y_true, y_pred, pos_label=1)
    f1_1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Average metrics
    precision_avg = precision_score(y_true, y_pred, average='macro')
    recall_avg = recall_score(y_true, y_pred, average='macro')
    f1_avg = f1_score(y_true, y_pred, average='macro')
    
    # Create a dictionary with all metrics
    metrics_dict = {
        'model': model_name,
        'accuracy': accuracy,
        'precision_class0': precision_0,
        'recall_class0': recall_0,
        'f1_class0': f1_0,
        'precision_class1': precision_1,
        'recall_class1': recall_1,
        'f1_class1': f1_1,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg
    }
    
    # Check if file exists
    if os.path.exists(filepath):
        # Read existing data and append new row
        metrics_df = pd.read_csv(filepath)
        
        # Check if model already exists in the dataframe
        if model_name in metrics_df['model'].values:
            # Update existing row
            metrics_df.loc[metrics_df['model'] == model_name] = pd.Series(metrics_dict)
        else:
            # Append new row
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_dict])], ignore_index=True)
    else:
        # Create new dataframe
        metrics_df = pd.DataFrame([metrics_dict])
    
    # Save to CSV
    metrics_df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
    
    return metrics_df

# Example usage:
if __name__ == "__main__":
    # Load data
    # Assume train_set_encoded.csv and test_set_encoded.csv are loaded
    train_data = pd.read_csv('../X_train_encoded.csv')
    test_data = pd.read_csv('../X_test_encoded.csv')
    
    # Split features and target
    X_train = train_data.drop('Depression', axis=1)
    y_train = train_data['Depression']
    
    # Split train into train and validation
    X_train_encoded, X_valid_encoded, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    X_test_encoded = test_data.drop('Depression', axis=1)
    y_test = test_data['Depression']
    
    # Convert to numpy arrays
    X_train_np = X_train_encoded.values.astype(float)
    X_valid_np = X_valid_encoded.values.astype(float)
    X_test_np = X_test_encoded.values.astype(float)
    
    y_train_np = y_train.values.astype(int)
    y_valid_np = y_valid.values.astype(int)
    y_test_np = y_test.values.astype(int)
    
    # Add bias term
    ones_train = np.ones((X_train_np.shape[0], 1), dtype=float)
    ones_valid = np.ones((X_valid_np.shape[0], 1), dtype=float)
    ones_test = np.ones((X_test_np.shape[0], 1), dtype=float)
    
    X_train_np = np.hstack([ones_train, X_train_np])
    X_valid_np = np.hstack([ones_valid, X_valid_np])
    X_test_np = np.hstack([ones_test, X_test_np])
    
    # Create and train model
    model = LogisticRegression(lr=0.01, epochs=5000, C=0.1, threshold=0.5)
    model.fit(X_train_np, y_train_np)
    
    # Make predictions
    y_train_pred = model.predict(X_train_np)
    y_valid_pred = model.predict(X_valid_np)
    y_test_pred = model.predict(X_test_np)
    
    # Evaluate model
    train_accuracy = np.mean(y_train_pred == y_train_np)
    valid_accuracy = np.mean(y_valid_pred == y_valid_np)
    test_accuracy = np.mean(y_test_pred == y_test_np)
    
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {valid_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Print classification report
    print("\nClassification Report (Train):")
    print(classification_report(y_train_np, y_train_pred))
    print("\nClassification Report (Validation):")
    print(classification_report(y_valid_np, y_valid_pred, digits=4))
    print("\nClassification Report (Test):")
    print(classification_report(y_test_np, y_test_pred, digits=4))
    
    # Save metrics to CSV file
    save_metrics_to_csv("LogisticRegression_Scratch", y_test_np, y_test_pred)
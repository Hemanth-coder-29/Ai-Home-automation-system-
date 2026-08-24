import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# Define file paths
CSV_FILE = 'dataset/gesture_dataset.csv'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.keras')

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 1. Load Dataset
print("Loading dataset...")
data = pd.read_csv(CSV_FILE)

# Separate features (X) and labels (y)
X = data.drop('label', axis=1).values
y = data['label'].values

# 2. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build Multi-Layer Perceptron (MLP) Architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')  # 4 classes: 0, 1, 2, 3
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train Model
print("Training AI model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# 5. Evaluate Performance
print("\n--- MODEL EVALUATION ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Open Palm', 'Fist', 'Victory', 'Thumbs Up']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# 6. Save Model
model.save(MODEL_PATH)
print(f"\nModel saved successfully to {MODEL_PATH}")
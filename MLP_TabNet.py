import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Bitcoin transaction dataset used for ransomware detection
btc_data = pd.read_csv("BitcoinHeistData.csv")
btc_data.columns = btc_data.columns.str.strip().str.lower()
print("Columns:", btc_data.columns.tolist())

print("Unique ransomware families:", btc_data["label"].nunique())
print("Initial dataset size:", len(btc_data))

# Remove ransomware families with extremely low samples since they cannot
# be reliably learned or split during training/testing
label_counts = btc_data["label"].value_counts()
valid_labels = label_counts[label_counts > 1].index

btc_data = btc_data[btc_data["label"].isin(valid_labels)]

# Apply stratified sampling to maintain representation of each ransomware
# family while reducing dataset size for faster experimentation
btc_data = btc_data.groupby("label", group_keys=False).sample(
    n=6000, replace=True, random_state=42
).reset_index(drop=True)

# Basic validation to check for missing values in transaction data
if btc_data.isnull().sum().sum() > 0:
    print("Warning: Missing values detected in dataset")

# Drop non-useful column if it exists (like address)
if "address" in btc_data.columns:
    btc_data = btc_data.drop("address", axis=1)

# Separate transaction_features and label
transaction_features = btc_data.drop(columns=["label"])
ransomware_labels = btc_data["label"]

# Encode ransomware_labels (strings -> numbers)
le = LabelEncoder()
ransomware_labels = le.fit_transform(ransomware_labels)

# Normalize transaction-based features to prevent large-value attributes
# from dominating the learning process in the neural network
scaler = StandardScaler()
transaction_features = scaler.fit_transform(transaction_features)

# Split dataset into training and testing sets while preserving class distribution
X_train, X_test, y_train, y_test = train_test_split(
    transaction_features, ransomware_labels, test_size=0.2, random_state=42, stratify=ransomware_labels
)

print("Preprocessing complete!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Dataset prepared. Beginning training for MLP and TabNet models...")

# Building the MLP
import torch
import torch.nn as nn
import torch.optim as optim

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

class MLP(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.mlp_model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.mlp_model(x)
    
# Initalize
mlp_model = MLP(X_train.shape[1], len(set(ransomware_labels)))
optimizer = optim.Adam(mlp_model.parameters(), lr=0.0005)

# Use weighted loss to address class imbalance in ransomeware categories
import numpy as np

# Compute class weights
class_counts = np.bincount(y_train)

# Avoid division by zero
class_counts[class_counts == 0] = 1

class_weights = 1. / np.sqrt(class_counts)
weights = torch.FloatTensor(class_weights)

criterion = nn.CrossEntropyLoss(weight=weights)

# Train the MLP mlp_model using mini-batches to improve learning stability
from torch.utils.data import TensorDataset, DataLoader

# Create dataset + loader
btc_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
btc_train_loader = DataLoader(btc_train_dataset, batch_size=64, shuffle=True)

# Train with batches
for epoch in range(20):
    total_loss = 0

    for batch_X, batch_y in btc_train_loader:
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(btc_train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# Evaluate mlp_model performance using accuracy and classification metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get predictions
with torch.no_grad():
    outputs = mlp_model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# Convert to numpy
y_pred = predicted.numpy()

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nMLP Accuracy:", acc)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# TabNet
from pytorch_tabnet.tab_model import TabNetClassifier

print("\nTraining Tabnet...")

tabnet = TabNetClassifier(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_params=dict(lr=2e-2),
)

tabnet.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=20,
    patience=5
)

# Predictions
tabnet_preds = tabnet.predict(X_test)

from sklearn.metrics import accuracy_score

print("\nTabnet Accuracy:", accuracy_score(y_test, tabnet_preds))

print("\nTabNet Classification Report:")
print(classification_report(y_test, tabnet_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tabnet_preds))


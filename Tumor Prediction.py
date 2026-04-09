import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pennylane as qml

# -----------------------------
# 1. Create results folder
# -----------------------------
os.makedirs("results", exist_ok=True)

# -----------------------------
# 2. Load dataset
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Class names:", data.target_names)

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Reduce dimensions for quantum circuit
#    We use 4 components = 4 qubits
# -----------------------------
n_qubits = 4
pca = PCA(n_components=n_qubits)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Reduced training shape:", X_train_pca.shape)
print("Reduced testing shape:", X_test_pca.shape)

# -----------------------------
# 6. Convert data to torch tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# -----------------------------
# 7. Define quantum device
# -----------------------------
dev = qml.device("default.qubit", wires=n_qubits)

# -----------------------------
# 8. Define quantum circuit
# -----------------------------
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Shape of trainable weights inside the quantum circuit
weight_shapes = {"weights": (2, n_qubits, 3)}

# Convert quantum circuit to Torch layer
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# -----------------------------
# 9. Build hybrid model
# -----------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_qubits, n_qubits)
        self.relu = nn.ReLU()
        self.quantum = quantum_layer
        self.fc2 = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.quantum(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = HybridModel()
print(model)

# -----------------------------
# 10. Loss and optimizer
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 11. Training loop
# -----------------------------
epochs = 30
batch_size = 16
loss_history = []

for epoch in range(epochs):
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (X_train_tensor.size(0) / batch_size)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# -----------------------------
# 12. Evaluate model
# -----------------------------
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred = (y_pred_probs >= 0.5).float()

accuracy = accuracy_score(y_test, y_pred.numpy())
print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred.numpy(), target_names=data.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred.numpy()))

# -----------------------------
# 13. Plot training loss
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/training_loss.png")
plt.show()

# -----------------------------
# 14. Plot PCA projection of dataset
#    (for visualization only, using first 2 PCA components)
# -----------------------------
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap="coolwarm", alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Breast Cancer Dataset (2D PCA Projection)")
plt.colorbar(scatter, ticks=[0, 1], label="Class")
plt.tight_layout()
plt.savefig("results/dataset_pca_plot.png")
plt.show()

# -----------------------------
# 15. Save results to text file
# -----------------------------
with open("results/model_results.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred.numpy(), target_names=data.target_names))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred.numpy())))

print("\nAll results saved in the 'results' folder.")
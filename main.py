import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from src.train import train_model, evaluate, plot_confusion_matrix

from src.data_loader import load_data
from src.graph_builder import build_graph
from src.model import GCN, GAT
from src.utils import create_masks
from src.train import train_model, evaluate

# ----------------------------
# 1. Load Data
# ----------------------------
print("Loading dataset...")
X, y = load_data("data/oasis_cross-sectional.csv")

# ----------------------------
# 2. Build Graph
# ----------------------------
print("Building graph...")
edge_index = build_graph(X)

# ----------------------------
# 3. Convert to PyG format
# ----------------------------
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# ----------------------------
# 4. Train-Test Split
# ----------------------------
train_mask, test_mask = create_masks(y)
data.train_mask = train_mask
data.test_mask = test_mask

# ----------------------------
# 5. Train GCN (Baseline)
# ----------------------------
print("\n--- Training GCN (Baseline) ---")
gcn = GCN(input_dim=x.shape[1])
gcn, gcn_losses = train_model(gcn, data, epochs=100)

gcn_acc = evaluate(gcn, data)
print(f"GCN Accuracy: {gcn_acc:.4f}")

# ----------------------------
# 6. Train GAT (Improved)
# ----------------------------
print("\n--- Training GAT (Improved) ---")
gat = GAT(input_dim=x.shape[1])
gat, gat_losses = train_model(gat, data, epochs=100)

gat_acc = evaluate(gat, data)
print(f"GAT Accuracy: {gat_acc:.4f}")

# ----------------------------
# 10. Confusion Matrices
# ----------------------------
print("\nGenerating Confusion Matrices...")

plot_confusion_matrix(gcn, data, 
                      title="GCN Confusion Matrix", 
                      filename="gcn_confusion_matrix.png")

plot_confusion_matrix(gat, data, 
                      title="GAT Confusion Matrix", 
                      filename="gat_confusion_matrix.png")

# ----------------------------
# 7. Plot Loss Comparison
# ----------------------------
plt.figure()
plt.plot(gcn_losses, label="GCN")
plt.plot(gat_losses, label="GAT")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 8. Plot Accuracy Comparison
# ----------------------------
plt.figure()
models = ["GCN", "GAT"]
accuracies = [gcn_acc, gat_acc]

plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 9. Save Figures (for PPT)
# ----------------------------
plt.figure()
plt.plot(gcn_losses, label="GCN")
plt.plot(gat_losses, label="GAT")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")

plt.figure()
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.grid(True)
plt.savefig("accuracy_comparison.png")

print("\n✅ Graphs saved as:")
print(" - loss_curve.png")
print(" - accuracy_comparison.png")
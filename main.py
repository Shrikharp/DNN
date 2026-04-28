import random
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.graph_builder import build_graph
from src.model import GCN, GAT
from src.utils import create_masks
from src.train import train_model, evaluate, evaluate_full, plot_confusion_matrix


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# 1. Load Data
# ----------------------------
print("Loading dataset...")
X, y = load_data("data/oasis_combined_clean.csv")

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
# 4. Train-Validation-Test Split
# ----------------------------
train_mask, val_mask, test_mask = create_masks(y)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print(f"Train samples: {train_mask.sum().item()}")
print(f"Validation samples: {val_mask.sum().item()}")
print(f"Test samples: {test_mask.sum().item()}")

# ----------------------------
# 5. Train GCN (Baseline)
# ----------------------------
print("\n--- Training GCN (Baseline) ---")
gcn = GCN(input_dim=x.shape[1], hidden_dim=16, dropout=0.2)
gcn, gcn_losses = train_model(
    gcn,
    data,
    epochs=100,
    lr=0.01,
    weight_decay=5e-4
)

gcn_acc = evaluate(gcn, data)
print(f"GCN Accuracy: {gcn_acc:.4f}")

# ----------------------------
# 6. Train GAT (Improved)
# ----------------------------
print("\n--- Training GAT (Improved) ---")
gat = GAT(input_dim=x.shape[1], hidden_dim=32, heads=4, dropout=0.2)
gat, gat_losses = train_model(
    gat,
    data,
    epochs=100,
    lr=0.01,
    weight_decay=5e-4
)

gat_acc = evaluate(gat, data)
print(f"GAT Accuracy: {gat_acc:.4f}")

print("\nGCN Detailed Evaluation:")
evaluate_full(gcn, data)

print("\nGAT Detailed Evaluation:")
evaluate_full(gat, data)
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
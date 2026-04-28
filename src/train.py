from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm

def train_model(
    model,
    data,
    epochs=100,
    lr=0.01,
    weight_decay=5e-4
):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    train_losses = []
    val_losses = []

    progress_bar = tqdm(range(epochs), desc="Training")

    class_counts = torch.bincount(data.y[data.train_mask])
    class_weights = class_counts.sum() / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(data.y.device)

    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    for epoch in progress_bar:
        # ----------------------------
        # Training step
        # ----------------------------
        model.train()
        optimizer.zero_grad()

        out = model(data)

        train_loss = F.nll_loss(
            out[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights
        )

        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # ----------------------------
        # Validation monitoring only
        # ----------------------------
        model.eval()
        with torch.no_grad():
            val_out = model(data)

            val_loss = F.nll_loss(
                val_out[data.val_mask],
                data.y[data.val_mask]
            )

        val_losses.append(val_loss.item())

        progress_bar.set_postfix({
            "train_loss": f"{train_loss.item():.4f}",
            "val_loss": f"{val_loss.item():.4f}"
        })

    return model, train_losses

def evaluate(model, data):
    model.eval()
    
    out = model(data)
    _, pred = out.max(dim=1)

    correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    total = int(data.test_mask.sum())

    accuracy = correct / total
    return accuracy

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_full(model, data):
    model.eval()

    out = model(data)

    # predictions
    pred = out.argmax(dim=1).detach().cpu().numpy()

    # probability for class 1: Impaired
    prob = out.exp()[:, 1].detach().cpu().numpy()

    # use only test nodes
    test_mask = data.test_mask.cpu().numpy()
    y_true = data.y.detach().cpu().numpy()[test_mask]
    y_pred = pred[test_mask]
    y_prob = prob[test_mask]

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Impaired"]
    ))

    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, data, title="Confusion Matrix", filename=None):
    model.eval()
    
    out = model(data)
    _, pred = out.max(dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Add numbers inside matrix
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.colorbar()
    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    plt.show()
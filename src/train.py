from tqdm import tqdm
import torch
import torch.nn as nn

def train_model(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    losses = []  # store losses

    progress_bar = tqdm(range(epochs), desc="Training", ncols=100)

    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # store loss

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return model, losses

def evaluate(model, data):
    model.eval()
    
    out = model(data)
    _, pred = out.max(dim=1)

    correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    total = int(data.test_mask.sum())

    accuracy = correct / total
    return accuracy

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
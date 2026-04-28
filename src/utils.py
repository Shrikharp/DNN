import torch
from sklearn.model_selection import train_test_split
import numpy as np

def create_masks(y, test_size=0.2):
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return train_mask, test_mask
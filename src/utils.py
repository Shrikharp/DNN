import torch
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from sklearn.model_selection import train_test_split


def create_masks(y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    num_nodes = len(y)
    indices = list(range(num_nodes))

    y_np = y.cpu().numpy() if hasattr(y, "cpu") else y

    # First split: train vs temp
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y_np,
        train_size=train_size,
        random_state=random_state,
        stratify=y_np
    )

    # Second split: temp into validation and test
    # Since val_size and test_size are both 0.15, split temp 50/50.
    val_relative_size = val_size / (val_size + test_size)

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_relative_size,
        random_state=random_state,
        stratify=y_temp
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask
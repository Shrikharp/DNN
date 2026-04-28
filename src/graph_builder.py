import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

def build_graph(X, threshold=0.8):
    sim_matrix = cosine_similarity(X)
    edges = []

    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            if i != j and sim_matrix[i][j] > threshold:
                edges.append([i, j])

    edge_index = np.array(edges).T
    return torch.tensor(edge_index, dtype=torch.long)
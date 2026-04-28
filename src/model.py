import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# Model 1: GCN (Baseline)
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=2, dropout=0.4):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Dropout only active during training
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Model 2: GAT (Improved)
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=2, heads=4, dropout=0.4):
        super(GAT, self).__init__()

        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            output_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)
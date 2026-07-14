import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    global_mean_pool,
)


# =====================================================
# GNN BRANCH
# =====================================================

class GNN(nn.Module):

    def __init__(
        self,
        hidden_dim=384,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        # 9 node features
        self.gat1 = GATConv(
            9,
            hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.sage1 = SAGEConv(
            hidden_dim * heads,
            hidden_dim,
        )

        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.sage2 = SAGEConv(
            hidden_dim,
            hidden_dim,
        )

        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = dropout

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.sage1(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.sage2(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        return x


# =====================================================
# COMPLETE MODEL
# =====================================================

class Model(nn.Module):

    def __init__(
        self,
        desc_dim=166,
        hidden_dim=384,
        heads=4,
        dropout=0.2,
    ):

        super().__init__()

        self.gnn = GNN(
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        # Descriptor branch
        self.desc_net = nn.Sequential(

            nn.Linear(desc_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
        )

        # Fusion network
        self.fc = nn.Sequential(

            nn.Linear(384 * 4 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(
        self,
        solute_graph,
        solvent_graph,
        descriptors,
    ):

        g1 = self.gnn(solute_graph)
        g2 = self.gnn(solvent_graph)

        # graph fusion
        graph_features = torch.cat(
            [
                g1,
                g2,
                g1 * g2,
                torch.abs(g1 - g2),
            ],
            dim=1,
        )

        desc_features = self.desc_net(descriptors)

        x = torch.cat(
            [
                graph_features,
                desc_features,
            ],
            dim=1,
        )

        return self.fc(x)

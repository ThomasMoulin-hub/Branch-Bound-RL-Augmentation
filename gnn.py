# gnn.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class BipartiteGNN(nn.Module):
    """
    GNN inspirée de Learn2CompareNodes :
    - embedding par nœud
    - 3 couches GCN
    - pooling séparé sur nœuds variables / contraintes
    - concat avec features globales B&B
    - tête qui produit des Q-values sur les actions

    API externe inchangée.
    """

    def __init__(self, var_in_channels, con_in_channels,
                 hidden_channels, num_actions, global_feat_size):
        super(BipartiteGNN, self).__init__()

        # On suppose une seule dimension de features pour tous les nœuds
        # (var_in_channels = dim des features retournées par milp_to_pyg_data)
        self.input_norm = nn.LayerNorm(var_in_channels)
        self.input_emb = nn.Linear(var_in_channels, hidden_channels)

        # Trois couches GCN
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # MLP pour les features globales de l'état B&B
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat_size, hidden_channels),
            nn.ReLU()
        )

        # Tête finale : [var_embed, con_embed, global_embed] -> Q-values
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_actions)
        )

    @staticmethod
    def _masked_mean(x, mask, batch):
        """
        Moyenne sur les nœuds où mask == 1, groupée par graph (batch).
        Implémentation pure PyTorch, sans torch_scatter.

        x    : [N, F]
        mask : [N] (bool ou 0/1)
        batch: [N] indices de graph (0..B-1)
        """
        if x.numel() == 0:
            return x.new_zeros(0, x.size(-1))

        if mask.dtype != torch.bool:
            mask = mask > 0.5
        mask_f = mask.float()

        batch = batch.to(x.device)
        mask_f = mask_f.to(x.device)

        num_graphs = int(batch.max().item()) + 1

        # Somme des features par graph
        sum_x = torch.zeros(
            num_graphs, x.size(1),
            device=x.device, dtype=x.dtype
        )
        sum_x.index_add_(0, batch, x * mask_f.unsqueeze(-1))

        # Nombre de nœuds utilisés par graph
        count = torch.zeros(
            num_graphs,
            device=x.device,
            dtype=x.dtype
        )
        count.index_add_(0, batch, mask_f)
        count = count.clamp(min=1.0).unsqueeze(-1)

        return sum_x / count

    def forward(self, x, edge_index, batch, global_features, edge_weight=None):
        # On suppose le layout venant de milp_to_pyg_data: [..., is_var, is_con]
        is_var = x[:, -2]
        is_con = x[:, -1]

        # Embedding + 3 GCN
        h = self.input_emb(self.input_norm(x))
        h = F.relu(self.conv1(h, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv3(h, edge_index, edge_weight=edge_weight))

        # Pooling séparé variables / contraintes
        var_embed = self._masked_mean(h, is_var, batch)  # [B, hidden]
        con_embed = self._masked_mean(h, is_con, batch)  # [B, hidden]
        graph_embed = torch.cat([var_embed, con_embed], dim=1)  # [B, 2*hidden]

        # Features globales B&B
        global_embed = self.global_mlp(global_features)  # [B, hidden]

        # Combine et produit les Q-values
        combined = torch.cat([graph_embed, global_embed], dim=1)  # [B, 3*hidden]
        q_values = self.head(combined)  # [B, num_actions]

        return q_values


def milp_to_pyg_data(instance):
    """
    Convertit (A, b, c, type) en un objet PyG Data.

    Layout des features nœud (dim = 6) :
        0: main_scalar   (cost pour vars, rhs pour cons)
        1: degree_nnz    (degré normalisé par nb de nnz)
        2: bias_norm     (contraintes seulement, 0 pour vars)
        3: obj_cos_sim   (contraintes seulement, 0 pour vars)
        4: is_var        (1 pour vars, 0 pour cons)
        5: is_con        (0 pour vars, 1 pour cons)

    Retourne aussi edge_weight aligné avec edge_index (deux sens).
    """
    A, c, b = instance['A'], instance['c'], instance['b']
    ptype = instance.get('type', 'cover')

    A = np.asarray(A)
    c = np.asarray(c).astype(np.float32)
    b = np.asarray(b).astype(np.float32)

    n_cons, n_vars = A.shape
    eps = 1e-9

    # convention de signe sur l'objectif
    cost_feat = (-c if ptype == 'packing' else c).astype(np.float32)

    # degrés (nombre de nnz) normalisés
    var_degree = (np.count_nonzero(A, axis=0) / max(n_cons, 1)).astype(np.float32)
    con_degree = (np.count_nonzero(A, axis=1) / max(n_vars, 1)).astype(np.float32)

    # normes de lignes pour normalisation
    row_abs_sum = (np.sum(np.abs(A), axis=1).astype(np.float32) + eps)
    row_l2 = (np.linalg.norm(A, axis=1).astype(np.float32) + eps)
    c_l2 = float(np.linalg.norm(c) + eps)

    # biais normalisé par la somme des coefficients de la contrainte
    bias_norm = (b / row_abs_sum).astype(np.float32)

    # cos-sim entre chaque ligne A_i et l'objectif c
    dot = (A @ c).astype(np.float32)
    obj_cos_sim = (dot / (row_l2 * c_l2)).astype(np.float32)

    # nœuds variables
    var_x = np.stack([
        cost_feat,                             # main_scalar
        var_degree,                            # degree_nnz
        np.zeros(n_vars, dtype=np.float32),    # bias_norm (vars)
        np.zeros(n_vars, dtype=np.float32),    # obj_cos_sim (vars)
        np.ones(n_vars, dtype=np.float32),     # is_var
        np.zeros(n_vars, dtype=np.float32),    # is_con
    ], axis=1)

    # nœuds contraintes
    con_x = np.stack([
        b,                                     # main_scalar (rhs)
        con_degree,                            # degree_nnz
        bias_norm,
        obj_cos_sim,
        np.zeros(n_cons, dtype=np.float32),    # is_var
        np.ones(n_cons, dtype=np.float32),     # is_con
    ], axis=1)

    x = torch.tensor(np.vstack([var_x, con_x]), dtype=torch.float)

    # edges bipartites à partir des nnz de A
    rows, cols = np.nonzero(A)
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)

    src_vc = cols
    dst_vc = rows + n_vars
    src_cv = rows + n_vars
    dst_cv = cols

    edge_index = torch.tensor(
        np.vstack([
            np.concatenate([src_vc, src_cv]),
            np.concatenate([dst_vc, dst_cv])
        ]),
        dtype=torch.long
    )

    # poids d'arêtes : A_ij normalisé par sum |A_i*|
    coef = A[rows, cols].astype(np.float32)
    w = (coef / row_abs_sum[rows]).astype(np.float32)

    # dupliquer les poids pour les arêtes inverses
    edge_weight = torch.tensor(np.concatenate([w, w]), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

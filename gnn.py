# gnn.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class BipartiteGNN(nn.Module):

    def __init__(self, var_in_channels, con_in_channels,
                 hidden_channels, num_actions, global_feat_size):
        super(BipartiteGNN, self).__init__()


        self.input_norm = nn.LayerNorm(var_in_channels)
        self.input_emb = nn.Linear(var_in_channels, hidden_channels)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat_size, hidden_channels),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_actions)
        )

    @staticmethod
    def _masked_mean(x, mask, batch):

        if x.numel() == 0:
            return x.new_zeros(0, x.size(-1))

        if mask.dtype != torch.bool:
            mask = mask > 0.5
        mask_f = mask.float()

        batch = batch.to(x.device)
        mask_f = mask_f.to(x.device)

        num_graphs = int(batch.max().item()) + 1


        sum_x = torch.zeros(
            num_graphs, x.size(1),
            device=x.device, dtype=x.dtype
        )
        sum_x.index_add_(0, batch, x * mask_f.unsqueeze(-1))


        count = torch.zeros(
            num_graphs,
            device=x.device,
            dtype=x.dtype
        )
        count.index_add_(0, batch, mask_f)
        count = count.clamp(min=1.0).unsqueeze(-1)

        return sum_x / count

    def forward(self, x, edge_index, batch, global_features, edge_weight=None):
        is_var = x[:, -2]
        is_con = x[:, -1]

        h = self.input_emb(self.input_norm(x))
        h = F.relu(self.conv1(h, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv3(h, edge_index, edge_weight=edge_weight))

        var_embed = self._masked_mean(h, is_var, batch)  # [B, hidden]
        con_embed = self._masked_mean(h, is_con, batch)  # [B, hidden]
        graph_embed = torch.cat([var_embed, con_embed], dim=1)  # [B, 2*hidden]


        global_embed = self.global_mlp(global_features)  # [B, hidden]


        combined = torch.cat([graph_embed, global_embed], dim=1)  # [B, 3*hidden]
        q_values = self.head(combined)  # [B, num_actions]

        return q_values


def milp_to_pyg_data(instance):

    A, c, b = instance['A'], instance['c'], instance['b']
    ptype = instance.get('type', 'cover')

    A = np.asarray(A)
    c = np.asarray(c).astype(np.float32)
    b = np.asarray(b).astype(np.float32)

    n_cons, n_vars = A.shape
    eps = 1e-9

 
    cost_feat = (-c if ptype == 'packing' else c).astype(np.float32)

    var_degree = (np.count_nonzero(A, axis=0) / max(n_cons, 1)).astype(np.float32)
    con_degree = (np.count_nonzero(A, axis=1) / max(n_vars, 1)).astype(np.float32)


    row_abs_sum = (np.sum(np.abs(A), axis=1).astype(np.float32) + eps)
    row_l2 = (np.linalg.norm(A, axis=1).astype(np.float32) + eps)
    c_l2 = float(np.linalg.norm(c) + eps)

    bias_norm = (b / row_abs_sum).astype(np.float32)

    dot = (A @ c).astype(np.float32)
    obj_cos_sim = (dot / (row_l2 * c_l2)).astype(np.float32)

    var_x = np.stack([
        cost_feat,                             # main_scalar
        var_degree,                            # degree_nnz
        np.zeros(n_vars, dtype=np.float32),    # bias_norm (vars)
        np.zeros(n_vars, dtype=np.float32),    # obj_cos_sim (vars)
        np.ones(n_vars, dtype=np.float32),     # is_var
        np.zeros(n_vars, dtype=np.float32),    # is_con
    ], axis=1)

    con_x = np.stack([
        b,                                     # main_scalar (rhs)
        con_degree,                            # degree_nnz
        bias_norm,
        obj_cos_sim,
        np.zeros(n_cons, dtype=np.float32),    # is_var
        np.ones(n_cons, dtype=np.float32),     # is_con
    ], axis=1)

    x = torch.tensor(np.vstack([var_x, con_x]), dtype=torch.float)

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

    coef = A[rows, cols].astype(np.float32)
    w = (coef / row_abs_sum[rows]).astype(np.float32)

    edge_weight = torch.tensor(np.concatenate([w, w]), dtype=torch.float)



    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

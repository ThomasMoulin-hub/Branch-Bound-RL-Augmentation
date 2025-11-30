import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class BipartiteGNN(nn.Module):
    """
    Encode the MILP as a bipartite graph (variables + constraints).
    We use a linear embedding 4 -> hidden_channels, then two GCN layers.
    """
    def __init__(self, var_in_channels, con_in_channels,
                 hidden_channels, num_actions, global_feat_size):
        super(BipartiteGNN, self).__init__()

        # Raw node input: 4 features per node (see milp_to_pyg_data)
        self.input_emb = nn.Linear(var_in_channels, hidden_channels)

        # Graph convolutions over the bipartite graph
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # MLP for global B&B features (e.g. depth, gap, fringe size)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat_size, hidden_channels),
            nn.ReLU()
        )

        # Final head: concatenated [graph_embed, global_embed] -> Q-values over actions
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_actions)
        )

    #def forward(self, x, edge_index, batch, global_features):
    #    # Node encoding + GCN
    #    x = F.relu(self.input_emb(x))
    #    x = F.relu(self.conv1(x, edge_index))
    #    x = F.relu(self.conv2(x, edge_index))
    def forward(self, x, edge_index, batch, global_features, edge_weight=None):
        x = F.relu(self.input_emb(x))
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))

        # Pool all nodes into one graph-level embedding
        graph_embed = global_mean_pool(x, batch)
        # Encode global B&B state
        global_embed = self.global_mlp(global_features)

        # Merge graph-level and global features, then predict Q(s,·)
        combined = torch.cat([graph_embed, global_embed], dim=1)
        q_values = self.head(combined)
        return q_values


#def milp_to_pyg_data(instance):
#    """
#    Convert (A, b, c, type) into a PyG Data object.
#
#    Variable nodes: [cost_feat, degree_normalized, is_var, is_con]
#    Constraint nodes: [rhs, degree_normalized, is_var, is_con]
#    """
#    A, c, b = instance['A'], instance['c'], instance['b']
#    ptype = instance.get('type', 'cover')
#
#    n_cons, n_vars = A.shape
#
#    # For packing: c = -profit for the LP solver, so we flip sign to expose profit as a feature
#    if ptype == 'packing':
#        cost_feat = -c
#    else:
#        cost_feat = c
#
#    # Normalized degrees for variables and constraints
#    var_degree = np.count_nonzero(A, axis=0) / max(n_cons, 1) #np.sum(A, axis=0)
#    con_degree = np.count_nonzero(A, axis=1) / max(n_vars, 1) #np.sum(A, axis=1)
#
#    # Variable node features: [cost, degree, is_var=1, is_con=0]
#    var_feats = np.stack([cost_feat, var_degree], axis=1)
#    # Constraint node features: [rhs, degree, is_var=0, is_con=1]
#    con_feats = np.stack([b, con_degree], axis=1)
#
#    var_feats_aug = np.hstack([
#        var_feats,
#        np.ones((n_vars, 1)),          # is_var
#        np.zeros((n_vars, 1))          # is_con
#    ])
#    con_feats_aug = np.hstack([
#        con_feats,
#        np.zeros((n_cons, 1)),         # is_var
#        np.ones((n_cons, 1))           # is_con
#    ])
#
#    # Stack variable and constraint nodes into a single node feature matrix
#    x = torch.tensor(
#        np.vstack([var_feats_aug, con_feats_aug]),
#        dtype=torch.float
#    )
#
#    # Build bipartite edges:
#    #   vars indexed [0 .. n_vars-1]
#    #   constraints indexed [n_vars .. n_vars+n_cons-1]
#    rows, cols = np.nonzero(A)
#    src_vc = cols
#    dst_vc = rows + n_vars
#    src_cv = rows + n_vars
#    dst_cv = cols
#
#    edge_index = torch.tensor(
#        np.vstack([
#            np.concatenate([src_vc, src_cv]),
#            np.concatenate([dst_vc, dst_cv])
#        ]),
#        dtype=torch.long
#    )
#
#    return Data(x=x, edge_index=edge_index)
def milp_to_pyg_data(instance):
    """
    Convert (A, b, c, type) into a PyG Data object.

    Node feature layout (dim = 6):
    idx 0: main_scalar   (cost for vars, rhs for cons)
    idx 1: degree_nnz    (nnz-count normalized)
    idx 2: bias_norm     (constraints only; 0 for vars)
    idx 3: obj_cos_sim   (constraints only; 0 for vars)
    idx 4: is_var
    idx 5: is_con

    Also returns edge_weight aligned with edge_index (both directions).
    """
    A, c, b = instance['A'], instance['c'], instance['b']
    ptype = instance.get('type', 'cover')

    A = np.asarray(A)
    c = np.asarray(c).astype(np.float32)
    b = np.asarray(b).astype(np.float32)

    n_cons, n_vars = A.shape
    eps = 1e-9

    # objective sign convention (same as your original)
    cost_feat = (-c if ptype == 'packing' else c).astype(np.float32)

    # Degrees by nnz-count
    var_degree = (np.count_nonzero(A, axis=0) / max(n_cons, 1)).astype(np.float32)
    con_degree = (np.count_nonzero(A, axis=1) / max(n_vars, 1)).astype(np.float32)

    # Row norms for normalization
    row_abs_sum = (np.sum(np.abs(A), axis=1).astype(np.float32) + eps)
    row_l2 = (np.linalg.norm(A, axis=1).astype(np.float32) + eps)
    c_l2 = float(np.linalg.norm(c) + eps)

    # Constraint bias normalized by row coefficients
    bias_norm = (b / row_abs_sum).astype(np.float32)

    # Cosine similarity between each constraint row A_i and objective c
    # <A_i, c> / (||A_i|| * ||c||)
    # A may be int; dot becomes float
    dot = (A @ c).astype(np.float32)
    obj_cos_sim = (dot / (row_l2 * c_l2)).astype(np.float32)

    # Build node features (6 dims)
    # Variables
    var_x = np.stack([
        cost_feat,          # main_scalar
        var_degree,         # degree
        np.zeros(n_vars, dtype=np.float32),  # bias_norm
        np.zeros(n_vars, dtype=np.float32),  # obj_cos_sim
        np.ones(n_vars, dtype=np.float32),   # is_var
        np.zeros(n_vars, dtype=np.float32)   # is_con
    ], axis=1)

    # Constraints
    con_x = np.stack([
        b,                  # main_scalar (rhs)
        con_degree,         # degree
        bias_norm,
        obj_cos_sim,
        np.zeros(n_cons, dtype=np.float32),  # is_var
        np.ones(n_cons, dtype=np.float32)    # is_con
    ], axis=1)

    x = torch.tensor(np.vstack([var_x, con_x]), dtype=torch.float)

    # Build bipartite edges from nonzeros in A
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

    # Edge weights: normalize A_ij by sum |A_i*| (per constraint row)
    coef = A[rows, cols].astype(np.float32)
    w = (coef / row_abs_sum[rows]).astype(np.float32)

    # duplicate weights for reverse edges
    edge_weight = torch.tensor(np.concatenate([w, w]), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


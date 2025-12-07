import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from gnn import BipartiteGNN


class PolicyGradientAgent:
    """
    REINFORCE with baseline (actor-critic style).
    Actor = GNN(policy)
    Critic = GNN(features)
    """

    def __init__(self, input_dim, hidden_dim, action_dim, global_dim,
                 lr=2e-5, gamma=0.95):
        # --- Actor (policy) ---
        self.policy = BipartiteGNN(
            var_in_channels=input_dim,
            con_in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_actions=action_dim,
            global_feat_size=global_dim,
        )

        # --- Critic (baseline V(s)) ---
        self.baseline = BipartiteGNN(
            var_in_channels=input_dim,
            con_in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_actions=action_dim,
            global_feat_size=global_dim,
            scalar_output=True
        )

        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_baseline = torch.optim.Adam(self.baseline.parameters(), lr=lr)

        self.gamma = gamma
        self.action_dim = action_dim

    def get_action(self, pyg_data, global_feats):
        """
        Forward pass:
        - policy GNN gives logits over actions
        - critic GNN gives V(s)
        """
        self.policy.eval()
        self.baseline.eval()

        glob_t = torch.from_numpy(global_feats).float().unsqueeze(0)
        batch_idx = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
        edge_w = getattr(pyg_data, "edge_weight", None)

        with torch.no_grad():
            logits = self.policy(
                pyg_data.x, pyg_data.edge_index, batch_idx,
                glob_t, edge_weight=edge_w,
            )  # [1, A]

        # Sample action from distribution defined by logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item()
    
    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def update(self, states, actions, values):
        """
        Compute REINFORCE loss with baseline:
            advantage = Gt - V(s)
            actor_loss = -log_pi(a|s) * advantage
            critic_loss = mse(V(s), Gt)

        PARAMS
            states: numpy array (states)
            actions: numpy array (actions)
            Qs: numpy array (Q values)
        """
        batch_size = len(states)
        actions = torch.LongTensor(actions)
        values = torch.FloatTensor(values)

        total_loss_baseline = 0.0
        total_loss_policy = 0.0

        # First, for each s in states, COMPUTE:
        # 1. value predictions, using value network
        # 2. probability vector pi(s), using policy network
        # Could probably put code elsewhere for cleanliness

        # Use padding to correct for variable trajectory lengths
        max_len = max(len(traj) for traj in states)  # for padding short trajectories
        padded_Vpreds = []
        padded_logits = []
        mask = []

        for traj in states:
            datas = [d for (d, _) in traj]
            batched_data = Batch.from_data_list(datas)

            #node_feats = torch.zeros(batched_data.num_nodes, dtype=torch.long)
            batch_idx = batched_data.batch

            global_feats = torch.stack([
                torch.from_numpy(vec).float() for (_, vec) in traj
            ], dim=0)  #[n_traj, n_global_feats]

            vals = self.baseline(
                batched_data.x,
                batched_data.edge_index,
                batch_idx,
                global_feats,
                getattr(batched_data, "edge_weight", None)
            ) # scalar

            logits = self.policy(
                batched_data.x,
                batched_data.edge_index,
                batch_idx,
                global_feats,
                getattr(batched_data, "edge_weight", None)
            ) # [1, A]

            # pad to max_len
            pad_size = max_len - len(traj)
            if pad_size > 0:
                vals = torch.cat([
                    vals,
                    torch.zeros(pad_size, vals.size(-1))
                ])
                logits = torch.cat([
                    logits,
                    torch.zeros(pad_size, logits.size(-1))
                ])

            padded_Vpreds.append(vals)
            padded_logits.append(logits)
            mask.append([1]*len(traj) + [0]*pad_size)

        # 1. V_preds
        V_preds = torch.stack(padded_Vpreds)  #[n_trajs, max_len]
        logits = torch.stack(padded_logits)  #[n_trajs, max_len, action_dim]
        mask = torch.tensor(mask, dtype=bool)  #[n_trajs, max_len]

        # Flatten predictions using the mask
        # This removes padding and aligns the shape with flat 'values' and 'actions' arrays
        # flat_V_preds shape will be [Total_Steps, 1]
        flat_V_preds = V_preds[mask] 

        # Ensure targets are the right shape
        # values shape: [Total_Steps] -> [Total_Steps, 1]
        flat_targets = values.view(-1, 1).to(flat_V_preds.device)

        # 2. probs
        logits = logits.masked_fill(~mask.unsqueeze(-1), float('-inf')) # mask padded steps
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Select only valid steps for the policy update
        flat_probs = probs[mask] # [Total_Steps, action_dim]

        # --- Train baseline ---
        # Use the batch (OBS, VAL) of states and value predictions as targets to train baseline.
        self.baseline.train()

        # Set LOSS as square error of predicted values compared to targets
        loss_baseline = torch.nn.functional.mse_loss(flat_V_preds, flat_targets)
        
        self.optimizer_baseline.zero_grad()
        loss_baseline.backward()
        nn.utils.clip_grad_norm_(self.baseline.parameters(), max_norm=1.0) #clip gradient to avoid explosions
        self.optimizer_baseline.step()

        total_loss_baseline += loss_baseline.item()

        # --- Update the policy ---
        # Compute advantages using values and computed baselines
        # Update policy using policy.update using state, actions, and values
        self.policy.train()

        # Calculate advantage on the flat, valid tensors
        # Detach V_preds so gradients don't flow back into critic from actor loss
        advantages = flat_targets - flat_V_preds.detach()

        # Compute probability pi(s,a)
        # Assuming 'actions' is a flat array of shape [Total_Steps]
        actions_tensor = actions.to(flat_probs.device)
        action_onehot = self._to_one_hot(actions_tensor, self.action_dim)

        # Calculate selected probability on flat tensors
        prob_selected = torch.sum(flat_probs * action_onehot, axis=-1)

        # FOR ROBUSTNESS
        prob_selected += 1e-8

        # Define loss function as described in the text above
        loss_policy = -torch.mean(advantages.squeeze() * torch.log(prob_selected))

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0) #clip gradient to avoid explosions
        self.optimizer_policy.step()

        total_loss_policy += loss_policy.item()

        return total_loss_policy / batch_size
import numpy as np
import torch
import torch.nn as nn

from gnn import BipartiteGNN


class DQNAgent:
    def __init__(
        self,
        input_dim,
        hidden_dim,
        action_dim,
        global_dim,
        lr=2e-5,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9975,
        target_update_freq=50,
    ):
        self.model = BipartiteGNN(
            var_in_channels=input_dim,
            con_in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_actions=action_dim,
            global_feat_size=global_dim,
        )
        self.target_model = BipartiteGNN(
            var_in_channels=input_dim,
            con_in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_actions=action_dim,
            global_feat_size=global_dim,
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.gamma = gamma
        self.action_dim = action_dim

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update_freq = target_update_freq
        self.update_counter = 0

    def get_action(self, pyg_data, global_feats, greedy=False):
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(0, self.action_dim)

        self.model.eval()
        with torch.no_grad():
            glob_t = torch.from_numpy(global_feats).float().unsqueeze(0)
            batch_idx = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
            edge_w = getattr(pyg_data, "edge_weight", None)

            q_values = self.model(
                pyg_data.x,
                pyg_data.edge_index,
                batch_idx,
                glob_t,
                edge_weight=edge_w,
            )  # [1, A]

        return torch.argmax(q_values, dim=1).item()

    def update(self, batch_data):
        if not batch_data:
            return 0.0

        self.model.train()
        total_loss = 0.0

        for (pyg, glob, act, rew, next_pyg, next_glob, done) in batch_data:
            glob_t = torch.from_numpy(glob).float().unsqueeze(0)       # [1, G]
            next_glob_t = torch.from_numpy(next_glob).float().unsqueeze(0)

            batch_idx = torch.zeros(pyg.num_nodes, dtype=torch.long)
            edge_w = getattr(pyg, "edge_weight", None)

            q_values = self.model(
                pyg.x,
                pyg.edge_index,
                batch_idx,
                glob_t,
                edge_weight=edge_w,
            )  # [1, A]
            current_q = q_values[0, act]

            target_q = rew
            if not done:
                with torch.no_grad():
                    next_edge_w = getattr(next_pyg, "edge_weight", None)
                    next_q = self.target_model(
                        next_pyg.x,
                        next_pyg.edge_index,
                        batch_idx,
                        next_glob_t,
                        edge_weight=next_edge_w,
                    )  # [1, A]
                    target_q += self.gamma * torch.max(next_q).item()

            target_tensor = torch.tensor(target_q, dtype=torch.float)

            loss = self.criterion(current_q, target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())


        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return total_loss / len(batch_data)

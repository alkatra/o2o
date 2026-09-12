"""The goal-conditioned stochastic imitation-learning policy.

Both reported variants use this one class:

  goal-conditioned : real local goal vector, goal_valid = 1 at every step
  goal-zeroed      : goal vector forced to (0, 0), goal_valid = 0 at every step

The explicit validity flag matters because (0, 0) is already a meaningful goal vector in this
encoding -- it is the vector to the agent's own destination, which shrinks to near-zero as any
trajectory completes (4-9% of training timesteps across the five leave-one-out training sets
have goal magnitude below 0.1 m). Without the flag, "no destination given" would be
indistinguishable from "arrived". Carrying the flag in both variants keeps them
architecturally identical and identical in input dimensionality, so the ablation isolates the
goal signal rather than also changing the network.
"""
import torch
import torch.nn as nn

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class PointNetFOV(nn.Module):
    """Permutation-invariant encoder over the variable-length set of visible agents.

    Shared per-agent MLP (5 -> 32 -> 64 -> out_dim, ReLU) then masked max-pooling, so the
    embedding does not depend on how many agents happen to be visible.
    """

    def __init__(self, in_dim=5, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, fov_states, mask):
        # fov_states: (B, N, 5); mask: (B, N) with 1 = real agent, 0 = padding
        B, N, _ = fov_states.shape
        if N == 0:
            return torch.zeros(B, self.mlp[-1].out_features, device=fov_states.device)

        x = self.mlp(fov_states.reshape(B * N, -1)).reshape(B, N, -1)

        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(mask_expanded == 0, -1e9)
        x_max, _ = torch.max(x, dim=1)

        # Batch elements with no visible agent at all get a zero embedding, not -1e9.
        valid_batch_mask = (mask.sum(dim=1) > 0).unsqueeze(-1)
        return torch.where(valid_batch_mask, x_max, torch.zeros_like(x_max))


class StochasticFPVNavigationActor(nn.Module):
    """Egocentric goal-conditioned policy emitting a diagonal Gaussian over local velocity.

    Inputs per step: visible-agent set, 16-sector blind-spot LiDAR, local goal vector,
    (speed, heading change), goal validity flag, GRU hidden state.
    Outputs: mu (2,), log_std (2,) clamped to [-5, 2], new hidden state.
    """

    def __init__(self, lidar_bins=16, fov_dim=5, fov_embed_dim=64, lidar_embed_dim=32,
                 rnn_hidden_dim=256):
        super().__init__()

        self.fov_processor = PointNetFOV(in_dim=fov_dim, out_dim=fov_embed_dim)

        self.lidar_mlp = nn.Sequential(
            nn.Linear(lidar_bins, 32),
            nn.ReLU(),
            nn.Linear(32, lidar_embed_dim),
            nn.ReLU(),
        )

        combined_dim = fov_embed_dim + lidar_embed_dim + 5  # 2 goal, 2 ego_vel, 1 goal_valid

        self.gru = nn.GRUCell(input_size=combined_dim, hidden_size=rnn_hidden_dim)

        self.action_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # [mu_x, mu_y, log_std_x, log_std_y]
        )

        self.rnn_hidden_dim = rnn_hidden_dim

    def forward(self, fov_states, fov_mask, lidar, goal, ego_vel, goal_valid, hidden_state):
        fov_embed = self.fov_processor(fov_states, fov_mask)
        lidar_embed = self.lidar_mlp(lidar)

        x = torch.cat([fov_embed, lidar_embed, goal, ego_vel, goal_valid], dim=1)
        new_hidden = self.gru(x, hidden_state)

        action_params = self.action_head(new_hidden)
        mu = action_params[:, 0:2]
        log_std = torch.clamp(action_params[:, 2:4], min=LOG_STD_MIN, max=LOG_STD_MAX)

        return mu, log_std, new_hidden

    def forward_sequence(self, fov_seq, fov_mask_seq, lidar_seq, goal_seq, ego_vel_seq,
                         goal_valid_seq):
        """Unrolls the policy over a padded (T, B, ...) batch, carrying the hidden state."""
        T, B = lidar_seq.shape[0], lidar_seq.shape[1]
        hidden_state = torch.zeros(B, self.rnn_hidden_dim, device=lidar_seq.device)

        mus, log_stds = [], []
        for t in range(T):
            mu, log_std, hidden_state = self(fov_seq[t], fov_mask_seq[t], lidar_seq[t],
                                             goal_seq[t], ego_vel_seq[t], goal_valid_seq[t],
                                             hidden_state)
            mus.append(mu)
            log_stds.append(log_std)

        return torch.stack(mus), torch.stack(log_stds)

"""Torch ``Dataset`` over the egocentric sequences written by ``build_fpv_dataset.py``.

Each item is one pedestrian's full egocentric sequence (variable length). Per timestep it
yields exactly the state the policy consumes and the action it is trained to imitate:

  fov_state : (N, 5)  per visible agent [local_x, local_y, local_vx, local_vy, frames_tracked]
  lidar     : (16,)   blind-spot polar depth, 16 sectors over 2*pi, capped at LIDAR_MAX_RANGE
  goal      : (2,)    ego's own destination in the ego frame
  ego_vel   : (2,)    (speed, heading change)
  action    : (2,)    next-step velocity in the ego frame (the imitation target)

Agents marked ``red`` by the exporter (occluded, or beyond its ray range) are not observed at
all and enter neither channel.
"""
import json
import math
import os

import torch
from torch.utils.data import Dataset

from transforms import global_to_local, generate_lidar

NUM_LIDAR_BINS = 16
LIDAR_MAX_RANGE = 15.0
TIMESTEP_SECONDS = 0.4
# ETH/UCY annotations step 10 raw frames per 0.4 s sample.
FRAME_STRIDE = 10
# Timesteps before the ego has this many samples of usable history are skipped.
HISTORY_LEN = 7


class FPVSequenceDataset(Dataset):
    def __init__(self, fpv_dir, raw_dir, num_lidar_bins=NUM_LIDAR_BINS,
                 max_lidar_range=LIDAR_MAX_RANGE, history_len=HISTORY_LEN, allowed_folders=None):
        """
        fpv_dir:         one FOV directory produced by build_fpv_dataset.py, e.g. <root>/135_deg
        raw_dir:         the matching raw ETH/UCY .txt directory
        allowed_folders: scene names to include (used to hold a split out for LOO training)
        """
        self.fpv_dir = fpv_dir
        self.raw_dir = raw_dir
        self.num_lidar_bins = num_lidar_bins
        self.max_lidar_range = max_lidar_range
        self.history_len = history_len
        self.allowed_folders = allowed_folders

        self.sequences = []
        self._load_data()

    def _load_data(self):
        for scene_name in sorted(os.listdir(self.fpv_dir)):
            if self.allowed_folders is not None and scene_name not in self.allowed_folders:
                continue

            scene_path = os.path.join(self.fpv_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            raw_file = os.path.join(self.raw_dir, f"{scene_name}.txt")
            if not os.path.exists(raw_file):
                print(f"warning: {raw_file} missing, skipping scene {scene_name}")
                continue

            for ego_file in sorted(os.listdir(scene_path)):
                if not ego_file.endswith(".json"):
                    continue

                with open(os.path.join(scene_path, ego_file), 'r') as f:
                    ego_data = json.load(f)
                if not ego_data:
                    continue

                ego_full_history = ego_data[0]["ego_agent"]["egofullhistory"]
                ego_pos_map = {p[0]: (p[1], p[2]) for p in ego_full_history}

                # Start supervising once the ego is actually walking, plus enough history for
                # the GRU to have context.
                first_moving_frame = ego_data[0]["frame_id"]
                for frame_data in ego_data:
                    if frame_data["ego_agent"]["egocurrentspeed"] > 0.3:
                        first_moving_frame = frame_data["frame_id"]
                        break

                seq = []
                for frame_data in ego_data:
                    if frame_data["frame_id"] < first_moving_frame + (self.history_len * FRAME_STRIDE):
                        continue
                    seq.append({"frame_data": frame_data, "ego_pos_map": ego_pos_map})

                if seq:
                    self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def _process_frame(self, seq_frame):
        frame_data = seq_frame["frame_data"]
        frame_id = int(frame_data["frame_id"])
        ego_pos_map = seq_frame["ego_pos_map"]

        ego = frame_data["ego_agent"]
        ego_x, ego_y = ego_pos_map.get(float(frame_id), (0.0, 0.0))

        prev_frame_id = frame_id - FRAME_STRIDE
        prev_prev_frame_id = frame_id - 2 * FRAME_STRIDE

        ego_speed = 0.0
        heading_change = 0.0

        if float(prev_frame_id) in ego_pos_map:
            prev_x, prev_y = ego_pos_map[float(prev_frame_id)]
            vel_x = (ego_x - prev_x) / TIMESTEP_SECONDS
            vel_y = (ego_y - prev_y) / TIMESTEP_SECONDS
            norm = math.hypot(vel_x, vel_y)
            ego_speed = norm

            if norm > 1e-3:
                heading_nx, heading_ny = vel_x / norm, vel_y / norm
            else:
                heading_nx, heading_ny = ego["egocurrentorientation"]

            if float(prev_prev_frame_id) in ego_pos_map:
                pprev_x, pprev_y = ego_pos_map[float(prev_prev_frame_id)]
                pvel_x = (prev_x - pprev_x) / TIMESTEP_SECONDS
                pvel_y = (prev_y - pprev_y) / TIMESTEP_SECONDS
                pnorm = math.hypot(pvel_x, pvel_y)
                if pnorm > 1e-3 and norm > 1e-3:
                    p_nx, p_ny = pvel_x / pnorm, pvel_y / pnorm
                    heading_change = math.atan2(heading_ny, heading_nx) - math.atan2(p_ny, p_nx)
                    while heading_change > math.pi:
                        heading_change -= 2 * math.pi
                    while heading_change < -math.pi:
                        heading_change += 2 * math.pi
        else:
            heading_nx, heading_ny = ego["egocurrentorientation"]

        norm = math.hypot(heading_nx, heading_ny)
        if norm > 0:
            heading_nx, heading_ny = heading_nx / norm, heading_ny / norm
        else:
            heading_nx, heading_ny = 1.0, 0.0

        # Imitation target: the velocity that carries the ego to its next annotated position.
        next_frame_id = frame_id + FRAME_STRIDE
        if float(next_frame_id) in ego_pos_map:
            next_x, next_y = ego_pos_map[float(next_frame_id)]
            global_target_vx = (next_x - ego_x) / TIMESTEP_SECONDS
            global_target_vy = (next_y - ego_y) / TIMESTEP_SECONDS
        else:
            global_target_vx = 0.0
            global_target_vy = 0.0

        _, _, local_action_vx, local_action_vy = global_to_local(
            0, 0, heading_nx, heading_ny, 0, 0, global_target_vx, global_target_vy)

        visible_agents = []
        blind_spot_obstacles = []

        for other in frame_data["other_agents"]:
            color = other["color"]
            if color == 'red':  # not observed this frame at all
                continue

            o_hist = other["past_5_history"]
            if not o_hist:
                continue

            o_x, o_y = o_hist[-1][1], o_hist[-1][2]
            o_vx, o_vy = other["orientation"]

            if color in ('green', 'yellow'):       # inside the FOV cone, line of sight clear
                lx, ly, lvx, lvy = global_to_local(ego_x, ego_y, heading_nx, heading_ny,
                                                   o_x, o_y, o_vx, o_vy)
                visible_agents.append([lx, ly, lvx, lvy, float(other["tracked_timesteps"])])
            elif color in ('gray', 'black'):       # outside the cone -> coarse channel only
                blind_spot_obstacles.append((o_x, o_y))

        lidar = generate_lidar(ego_x, ego_y, heading_nx, heading_ny, blind_spot_obstacles,
                               self.num_lidar_bins, self.max_lidar_range)

        goal_x, goal_y = ego["egogoal"]
        local_gx, local_gy, _, _ = global_to_local(ego_x, ego_y, heading_nx, heading_ny,
                                                   goal_x, goal_y, 0, 0)

        return {
            "fov_state": torch.tensor(visible_agents, dtype=torch.float32),
            "lidar": torch.tensor(lidar, dtype=torch.float32),
            "goal": torch.tensor([local_gx, local_gy], dtype=torch.float32),
            "ego_vel": torch.tensor([ego_speed, heading_change], dtype=torch.float32),
            "action": torch.tensor([local_action_vx, local_action_vy], dtype=torch.float32),
        }

    def __getitem__(self, idx):
        return [self._process_frame(f) for f in self.sequences[idx]]


def collate_seq_fn(batch):
    """Pads a batch of variable-length sequences over both time and visible-agent count.

    Returns (T, B, ...) tensors plus ``seq_mask`` marking real timesteps.
    """
    B = len(batch)
    lengths = [len(seq) for seq in batch]
    max_t = max(lengths)
    max_n = max((f["fov_state"].shape[0] for seq in batch for f in seq), default=0)

    padded_fov = torch.zeros((max_t, B, max_n, 5))
    padded_fov_mask = torch.zeros((max_t, B, max_n))
    padded_lidar = torch.zeros((max_t, B, NUM_LIDAR_BINS))
    padded_goal = torch.zeros((max_t, B, 2))
    padded_ego_vel = torch.zeros((max_t, B, 2))
    padded_action = torch.zeros((max_t, B, 2))
    seq_mask = torch.zeros((max_t, B))

    for b, seq in enumerate(batch):
        for t in range(lengths[b]):
            frame = seq[t]
            n = frame["fov_state"].shape[0]
            if n > 0:
                padded_fov[t, b, :n, :] = frame["fov_state"]
                padded_fov_mask[t, b, :n] = 1.0
            padded_lidar[t, b, :] = frame["lidar"]
            padded_goal[t, b, :] = frame["goal"]
            padded_ego_vel[t, b, :] = frame["ego_vel"]
            padded_action[t, b, :] = frame["action"]
            seq_mask[t, b] = 1.0

    return {
        "fov_state": padded_fov,
        "fov_mask": padded_fov_mask,
        "lidar": padded_lidar,
        "goal": padded_goal,
        "ego_vel": padded_ego_vel,
        "action": padded_action,
        "seq_mask": seq_mask,
        "lengths": lengths,
    }

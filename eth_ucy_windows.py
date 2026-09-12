"""Standard ETH/UCY evaluation windows (obs 8 / pred 12), self-contained.

Reproduces the window construction every trajectory-forecasting baseline on this benchmark
uses, so the reported numbers are comparable: slide a 20-frame window over the annotated
frames one frame at a time, keep the pedestrians present for the whole window, and keep the
window if at least two such pedestrians remain.

Coordinates are rounded to 4 decimals exactly as in the reference implementation, because the
per-agent destination lookup in ``evaluate.py`` keys on rounded positions.
"""
import math
import os

import numpy as np

OBS_LEN = 8
PRED_LEN = 12
MIN_PEDS_PER_WINDOW = 2


def read_file(path, delim='\t'):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delim) if delim in line else line.split()
            data.append([float(v) for v in parts])
    return np.asarray(data)


def load_windows(data_dir, obs_len=OBS_LEN, pred_len=PRED_LEN, skip=1, delim='\t'):
    """Returns a list of (obs_traj, pred_traj) with shapes (obs_len, N, 2) and (pred_len, N, 2)."""
    seq_len = obs_len + pred_len
    windows = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.txt'):
            continue
        data = read_file(os.path.join(data_dir, filename), delim)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = [data[data[:, 0] == frame, :] for frame in frames]
        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))

        for idx in range(0, num_sequences * skip + 1, skip):
            chunk = frame_data[idx:idx + seq_len]
            if len(chunk) < seq_len:
                continue
            curr_seq_data = np.concatenate(chunk, axis=0)

            kept = []
            for ped_id in np.unique(curr_seq_data[:, 1]):
                curr_ped_seq = np.around(curr_seq_data[curr_seq_data[:, 1] == ped_id, :],
                                         decimals=4)
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                # Present from the first to the last frame of the window...
                if pad_end - pad_front != seq_len:
                    continue
                # ...and actually sampled in every frame in between.
                if curr_ped_seq.shape[0] != seq_len:
                    continue
                kept.append(curr_ped_seq[:, 2:4])

            if len(kept) >= MIN_PEDS_PER_WINDOW:
                traj = np.stack(kept, axis=1)  # (seq_len, N, 2)
                windows.append((traj[:obs_len].astype(np.float32),
                                traj[obs_len:].astype(np.float32)))

    return windows


def build_coord_to_goal(data_dir, delim='\t'):
    """Maps every annotated (x, y) to that pedestrian's final annotated position.

    The evaluation protocol gives each agent its own destination, which ETH/UCY does not
    annotate; the end of the agent's complete recorded track is used as a stand-in, looked up
    from the agent's last observed position.
    """
    coord_to_goal = {}
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.txt'):
            continue
        data = read_file(os.path.join(data_dir, filename), delim)
        for ped_id in np.unique(data[:, 1]):
            ped_data = data[data[:, 1] == ped_id]
            fx, fy = ped_data[-1, 2], ped_data[-1, 3]
            for row in ped_data:
                coord_to_goal[(round(row[2], 4), round(row[3], 4))] = (fx, fy)
    return coord_to_goal

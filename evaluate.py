"""Stage 3: ADE/FDE evaluation under the self-play leave-one-out protocol.

Each held-out split is evaluated with the model that never trained on it. For every standard
obs-8/pred-12 window, *every* agent in the window is driven by an independent copy of the same
policy (shared weights, its own hidden state and its own goal), each computing its own
egocentric FOV/LiDAR state from the other agents' current positions. All agents step
synchronously, so from the second step on they are reacting to each other's model-produced
positions rather than to ground truth: a decentralised multi-agent system in closed loop, not
one agent against passive replayed others.

Reported per split:

  minADE20 / minFDE20   K=20 rollouts (rollout 0 = deterministic mean action, 1-19 sampled from
                        the learned Gaussian). The best rollout is selected per window by total
                        displacement summed over every agent and timestep, matching the
                        benchmark convention; ADE/FDE are then averaged over agent instances.
  K1 ADE / K1 FDE       the deterministic-mean rollout alone -- the single trajectory a deployed
                        policy would actually execute.
  mean-is-best          fraction of windows where the deterministic rollout is itself the
                        lowest-error member of the 20.

Usage:
    python evaluate.py --data-root data/eth-standard --checkpoint-dir checkpoints \
                       --variant goal_conditioned
    python evaluate.py --data-root data/eth-standard --checkpoint-dir checkpoints \
                       --variant goal_zeroed --split eth
    python evaluate.py --data-root data/eth-standard --variant orca      # analytic baseline
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from eth_ucy_windows import build_coord_to_goal, load_windows
from model import StochasticFPVNavigationActor
from transforms import generate_lidar, global_to_local, local_to_global_velocity

# Rollouts are one agent at a time on tiny tensors; intra-op threading costs more than it saves.
torch.set_num_threads(1)

SPLITS = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
NUM_SAMPLES = 20
DT = 0.4
FOV_DEG = 135.0
NUM_LIDAR_BINS = 16
LIDAR_MAX_RANGE = 15.0
DEFAULT_SEED = 20260908

# ORCA baseline parameters (neighbour distance, max neighbours, horizons, radius, max speed).
ORCA_PARAMS = dict(neighbor_dist=15.0, max_neighbors=10, time_horizon=5.0,
                   time_horizon_obst=5.0, radius=0.3, max_speed=1.5)


class Agent:
    def __init__(self, x, y, gx, gy, hx, hy, vx, vy):
        self.x, self.y = x, y
        self.gx, self.gy = gx, gy
        self.hx, self.hy = hx, hy
        self.prev_hx, self.prev_hy = hx, hy
        self.vx, self.vy = vx, vy
        self.hidden = torch.zeros(1, 256)
        self.tracked = {}


def init_agents(obs_traj, final_goals):
    """Seeds each agent's position, velocity and heading from the last two observed frames."""
    agents = []
    for i in range(obs_traj.shape[1]):
        x, y = float(obs_traj[-1, i, 0]), float(obs_traj[-1, i, 1])
        vx = (x - float(obs_traj[-2, i, 0])) / DT
        vy = (y - float(obs_traj[-2, i, 1])) / DT
        norm = math.hypot(vx, vy)
        hx, hy = (vx / norm, vy / norm) if norm > 1e-3 else (1.0, 0.0)

        a = Agent(x, y, final_goals[i][0], final_goals[i][1], hx, hy, vx, vy)

        p_vx = (float(obs_traj[-2, i, 0]) - float(obs_traj[-3, i, 0])) / DT
        p_vy = (float(obs_traj[-2, i, 1]) - float(obs_traj[-3, i, 1])) / DT
        p_norm = math.hypot(p_vx, p_vy)
        a.prev_hx, a.prev_hy = (p_vx / p_norm, p_vy / p_norm) if p_norm > 1e-3 else (hx, hy)

        agents.append(a)
    return agents


def rollout_policy(obs_traj, final_goals, model, pred_len, goal_conditioned, deterministic):
    """One self-play rollout of the whole window. Returns positions (pred_len, N, 2)."""
    agents = init_agents(obs_traj, final_goals)
    positions = np.zeros((pred_len, len(agents), 2))

    for t in range(pred_len):
        next_vxs, next_vys = [], []

        for i, a in enumerate(agents):
            visible, blind = [], []
            for j, b in enumerate(agents):
                if i == j:
                    continue
                lx, ly, lvx, lvy = global_to_local(a.x, a.y, a.hx, a.hy, b.x, b.y, b.vx, b.vy)
                rel_angle = math.atan2(ly, lx)
                if abs(rel_angle) <= math.radians(FOV_DEG / 2.0):
                    a.tracked[j] = a.tracked.get(j, 0) + 1
                    visible.append([lx, ly, lvx, lvy, float(a.tracked[j])])
                else:
                    a.tracked[j] = 0
                    blind.append((b.x, b.y))

            fov_t = (torch.tensor([visible], dtype=torch.float32) if visible
                     else torch.zeros(1, 0, 5))
            fov_mask_t = torch.ones(1, len(visible)) if visible else torch.zeros(1, 0)
            lidar_t = torch.from_numpy(generate_lidar(a.x, a.y, a.hx, a.hy, blind,
                                                      NUM_LIDAR_BINS, LIDAR_MAX_RANGE)).unsqueeze(0)

            ego_speed = math.hypot(a.vx, a.vy)
            heading_change = math.atan2(a.hy, a.hx) - math.atan2(a.prev_hy, a.prev_hx)
            while heading_change > math.pi:
                heading_change -= 2 * math.pi
            while heading_change < -math.pi:
                heading_change += 2 * math.pi
            ego_vel_t = torch.tensor([[ego_speed, heading_change]], dtype=torch.float32)

            if goal_conditioned:
                local_gx, local_gy, _, _ = global_to_local(a.x, a.y, a.hx, a.hy, a.gx, a.gy, 0, 0)
                goal_t = torch.tensor([[local_gx, local_gy]], dtype=torch.float32)
                flag_t = torch.ones(1, 1)
            else:
                goal_t = torch.zeros(1, 2)
                flag_t = torch.zeros(1, 1)

            with torch.no_grad():
                mu, log_std, a.hidden = model(fov_t, fov_mask_t, lidar_t, goal_t, ego_vel_t,
                                              flag_t, a.hidden)

            if deterministic:
                action = mu
            else:
                action = torch.distributions.Normal(mu, torch.exp(log_std)).sample()

            gvx, gvy = local_to_global_velocity(a.hx, a.hy, action[0, 0].item(), action[0, 1].item())
            next_vxs.append(gvx)
            next_vys.append(gvy)

        # Synchronous update: everyone moves on the state everyone else just saw.
        for i, a in enumerate(agents):
            a.vx, a.vy = next_vxs[i], next_vys[i]
            a.x += a.vx * DT
            a.y += a.vy * DT
            norm = math.hypot(a.vx, a.vy)
            a.prev_hx, a.prev_hy = a.hx, a.hy
            if norm > 1e-3:
                a.hx, a.hy = a.vx / norm, a.vy / norm
            positions[t, i] = [a.x, a.y]

    return positions


def rollout_orca(obs_traj, pred_traj_gt):
    """Reciprocal-velocity-obstacle baseline, steering straight at the true endpoint."""
    import rvo2

    num_agents = obs_traj.shape[1]
    pred_len = pred_traj_gt.shape[0]
    sim = rvo2.PyRVOSimulator(DT, ORCA_PARAMS['neighbor_dist'], ORCA_PARAMS['max_neighbors'],
                              ORCA_PARAMS['time_horizon'], ORCA_PARAMS['time_horizon_obst'],
                              ORCA_PARAMS['radius'], ORCA_PARAMS['max_speed'])

    ids, goals = [], []
    for i in range(num_agents):
        ids.append(sim.addAgent((float(obs_traj[-1, i, 0]), float(obs_traj[-1, i, 1]))))
        goals.append((float(pred_traj_gt[-1, i, 0]), float(pred_traj_gt[-1, i, 1])))

    positions = np.zeros((pred_len, num_agents, 2))
    for t in range(pred_len):
        for i in range(num_agents):
            gx, gy = goals[i]
            cx, cy = sim.getAgentPosition(ids[i])
            dist = math.hypot(gx - cx, gy - cy)
            pref = (0.0, 0.0) if dist < 0.1 else ((gx - cx) / dist * ORCA_PARAMS['max_speed'],
                                                  (gy - cy) / dist * ORCA_PARAMS['max_speed'])
            sim.setAgentPrefVelocity(ids[i], pref)
        sim.doStep()
        for i in range(num_agents):
            positions[t, i] = sim.getAgentPosition(ids[i])

    return positions


def per_agent_ade_fde(positions, gt):
    """ADE = mean displacement over the horizon, FDE = displacement at the final step."""
    d = np.linalg.norm(positions - gt, axis=-1)  # (T, N)
    return d.mean(axis=0), d[-1]


def evaluate_split(split, data_root, variant, checkpoint_dir, checkpoint=None, seed=DEFAULT_SEED):
    test_dir = os.path.join(data_root, split, 'test')
    windows = load_windows(test_dir)
    coord_to_goal = build_coord_to_goal(test_dir)

    model = None
    if variant != 'orca':
        path = checkpoint or os.path.join(checkpoint_dir, f"{variant}_{split}.pth")
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
        model = StochasticFPVNavigationActor()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        print(f"  {split}: {len(windows)} windows, checkpoint {os.path.basename(path)}", flush=True)
    else:
        print(f"  {split}: {len(windows)} windows, ORCA baseline", flush=True)

    best_ade, best_fde, k1_ade, k1_fde = [], [], [], []
    mean_is_best = 0

    for w_idx, (obs_traj, pred_traj_gt) in enumerate(windows):
        # Seed per window rather than once per split, so the sampled rollouts depend only on
        # the window index and not on how many windows were processed before it.
        torch.manual_seed(seed + w_idx)
        num_agents = obs_traj.shape[1]

        final_goals = []
        for i in range(num_agents):
            key = (round(float(obs_traj[-1, i, 0]), 4), round(float(obs_traj[-1, i, 1]), 4))
            fallback = (float(pred_traj_gt[-1, i, 0]), float(pred_traj_gt[-1, i, 1]))
            final_goals.append(coord_to_goal.get(key, fallback))

        if variant == 'orca':
            rollouts = [rollout_orca(obs_traj, pred_traj_gt)]
        else:
            goal_conditioned = (variant == 'goal_conditioned')
            rollouts = [rollout_policy(obs_traj, final_goals, model, pred_traj_gt.shape[0],
                                       goal_conditioned, deterministic=(k == 0))
                        for k in range(NUM_SAMPLES)]

        # Window-level best-of-K: one winning rollout for the whole window, chosen by the
        # displacement summed over every agent and every timestep.
        ade_sums = [float(np.linalg.norm(p - pred_traj_gt, axis=-1).sum()) for p in rollouts]
        winner = int(np.argmin(ade_sums))
        if winner == 0:
            mean_is_best += 1

        w_ade, w_fde = per_agent_ade_fde(rollouts[winner], pred_traj_gt)
        best_ade.extend(w_ade.tolist())
        best_fde.extend(w_fde.tolist())

        m_ade, m_fde = per_agent_ade_fde(rollouts[0], pred_traj_gt)
        k1_ade.extend(m_ade.tolist())
        k1_fde.extend(m_fde.tolist())

        if (w_idx + 1) % 100 == 0:
            print(f"    [{w_idx + 1}/{len(windows)} windows] running minADE "
                  f"{np.mean(best_ade):.4f}", flush=True)

    return {
        'minADE20': float(np.mean(best_ade)),
        'minFDE20': float(np.mean(best_fde)),
        'K1_ADE': float(np.mean(k1_ade)),
        'K1_FDE': float(np.mean(k1_fde)),
        'mean_is_best_of_20': mean_is_best / len(windows) if windows else float('nan'),
        'n_windows': len(windows),
        'n_agent_instances': len(best_ade),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True,
                    help='root holding <split>/test/*.txt for the five splits')
    ap.add_argument('--checkpoint-dir', default='checkpoints')
    ap.add_argument('--variant', choices=['goal_conditioned', 'goal_zeroed', 'orca'],
                    default='goal_conditioned')
    ap.add_argument('--checkpoint', default=None,
                    help='explicit checkpoint path (single --split only)')
    ap.add_argument('--split', choices=SPLITS, default=None, help='default: all five')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--out-json', default=None, help='also write results here')
    args = ap.parse_args()

    splits = [args.split] if args.split else SPLITS
    print(f"variant={args.variant} seed={args.seed} K={NUM_SAMPLES} obs=8 pred=12")

    results = {}
    for split in splits:
        results[split] = evaluate_split(split, args.data_root, args.variant,
                                        args.checkpoint_dir, args.checkpoint, args.seed)

    header = f"\n{'split':7} {'minADE20':>9} {'minFDE20':>9} {'K1 ADE':>8} {'K1 FDE':>8} {'mean-best':>10} {'agents':>8}"
    print(header)
    print('-' * len(header))
    for split in splits:
        r = results[split]
        print(f"{split:7} {r['minADE20']:9.4f} {r['minFDE20']:9.4f} {r['K1_ADE']:8.4f} "
              f"{r['K1_FDE']:8.4f} {r['mean_is_best_of_20'] * 100:9.1f}% {r['n_agent_instances']:8d}")

    if len(splits) > 1:
        avg = {k: float(np.mean([results[s][k] for s in splits]))
               for k in ('minADE20', 'minFDE20', 'K1_ADE', 'K1_FDE')}
        results['AVG'] = avg
        print(f"{'AVG':7} {avg['minADE20']:9.4f} {avg['minFDE20']:9.4f} {avg['K1_ADE']:8.4f} "
              f"{avg['K1_FDE']:8.4f}")

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == '__main__':
    main()

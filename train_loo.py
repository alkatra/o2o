"""Stage 2: leave-one-out training of the goal-conditioned and goal-zeroed policies.

For each of the five ETH/UCY splits, a model is trained on the egocentric sequences of the
other four and never sees the held-out split, so every evaluation number is reported on unseen
data.

Objective (all terms masked by the sequence mask, and all auxiliary terms evaluated on the
distribution mean so the centre behaviour is what they shape):

  NLL              Gaussian negative log-likelihood of the demonstrated next-step velocity
  speed range      x1.5   keeps predicted speed within one SD of the demonstrated speed
                          distribution (the slow side weighted twice, to stop it stalling)
  heading align    x1.0   negative cosine similarity between mean action and goal direction;
                          self-disables for the goal-zeroed variant, whose goal vector is (0,0)
  collision (TCPA) x0.05  time/distance-to-closest-approach potential over visible agents
  entropy bonus    x0.05  discourages premature variance collapse

Usage:
    python train_loo.py --fpv-dir fpv_datasets/eth-standard/135_deg \
                        --raw-dir data/eth-standard/raw/all_data \
                        --variant goal_conditioned --out-dir checkpoints
    python train_loo.py ... --variant goal_zeroed
    python train_loo.py ... --split zara1 --seed 20260908
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader

from fpv_dataset import FPVSequenceDataset, collate_seq_fn
from model import StochasticFPVNavigationActor

# Which raw scene folders make up each of the five standard ETH/UCY splits.
SPLIT_SCENES = {
    'eth': ['biwi_eth'],
    'hotel': ['biwi_hotel'],
    'univ': ['students001', 'students003', 'uni_examples'],
    'zara1': ['crowds_zara01'],
    'zara2': ['crowds_zara02'],
}

LAMBDA_SPEED = 1.5
LAMBDA_HEADING = 1.0
LAMBDA_COLLISION = 0.05
LAMBDA_ENTROPY = 0.05
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 1.0
TRAIN_FRACTION = 0.8


def calc_ttc_loss(mu, fov, fov_mask):
    """Time/distance-to-closest-approach penalty over the agents currently visible.

    For each visible agent, the closest approach the mean action would produce is computed
    analytically from relative position and velocity; only approaching pairs are penalised, the
    penalty grows as the closest-approach distance shrinks, and it is discounted by how far
    ahead in time that approach is.
    """
    if fov.size(2) == 0:
        return torch.zeros(mu.shape[0], mu.shape[1], device=mu.device)

    lx, ly = fov[:, :, :, 0], fov[:, :, :, 1]
    lvx, lvy = fov[:, :, :, 2], fov[:, :, :, 3]

    ego_vx = mu[:, :, 0].unsqueeze(2)
    ego_vy = mu[:, :, 1].unsqueeze(2)

    rel_vx = lvx - ego_vx
    rel_vy = lvy - ego_vy

    dot_prod = lx * rel_vx + ly * rel_vy
    v_sq = rel_vx ** 2 + rel_vy ** 2 + 1e-6
    tcpa = -dot_prod / v_sq

    approaching_mask = (dot_prod < 0).float()
    tcpa = torch.clamp(tcpa, min=0.0, max=5.0)  # only care about the next 5 seconds

    dcpa_x = lx + tcpa * rel_vx
    dcpa_y = ly + tcpa * rel_vy
    dcpa_sq = dcpa_x ** 2 + dcpa_y ** 2

    penalty = torch.exp(-tcpa) / (dcpa_sq + 1e-3)
    return (penalty * approaching_mask * fov_mask).sum(dim=2)


def make_goal_inputs(goal_real, variant, device):
    """goal_conditioned: real goal, flag 1. goal_zeroed: zeroed goal, flag 0."""
    T, B = goal_real.shape[0], goal_real.shape[1]
    if variant == 'goal_conditioned':
        return goal_real, torch.ones(T, B, 1, device=device)
    return torch.zeros_like(goal_real), torch.zeros(T, B, 1, device=device)


def batch_loss(model, batch, variant, device):
    fov_state = batch["fov_state"].to(device)
    fov_mask = batch["fov_mask"].to(device)
    lidar = batch["lidar"].to(device)
    ego_vel = batch["ego_vel"].to(device)
    action_gt = batch["action"].to(device)
    seq_mask = batch["seq_mask"].to(device)

    goal, goal_valid = make_goal_inputs(batch["goal"].to(device), variant, device)

    mu, log_std = model.forward_sequence(fov_state, fov_mask, lidar, goal, ego_vel, goal_valid)
    dist = Normal(mu, torch.exp(log_std))

    log_prob = dist.log_prob(action_gt).sum(dim=2)
    nll_loss = -(log_prob * seq_mask).sum() / seq_mask.sum()

    action_pred = mu

    pred_speed = torch.norm(action_pred, dim=2)
    gt_speed = torch.norm(action_gt, dim=2)
    valid_gt_speeds = gt_speed[seq_mask.bool()]
    if len(valid_gt_speeds) > 1:
        mean_speed = valid_gt_speeds.mean()
        std_speed = valid_gt_speeds.std()
        slow_penalty = 2.0 * torch.relu((mean_speed - std_speed) - pred_speed)
        fast_penalty = torch.relu(pred_speed - (mean_speed + std_speed))
        speed_range_loss = ((slow_penalty + fast_penalty) * seq_mask).sum() / seq_mask.sum()
    else:
        speed_range_loss = torch.tensor(0.0, device=device)

    # Zero goal vector => zero goal direction => cos_sim is 0 for every step, so this term
    # contributes nothing for the goal-zeroed variant without any special-casing.
    goal_dir = goal / (torch.norm(goal, dim=-1, keepdim=True) + 1e-6)
    mu_dir = action_pred / (torch.norm(action_pred, dim=-1, keepdim=True) + 1e-6)
    cos_sim = (mu_dir * goal_dir).sum(dim=-1)
    heading_loss = -((cos_sim * seq_mask).sum() / seq_mask.sum())

    ttc = calc_ttc_loss(action_pred, fov_state, fov_mask)
    collision_loss = (ttc * seq_mask).sum() / seq_mask.sum()

    entropy_loss = -(dist.entropy().sum(dim=2) * seq_mask).sum() / seq_mask.sum()

    total = (nll_loss
             + LAMBDA_SPEED * speed_range_loss
             + LAMBDA_HEADING * heading_loss
             + LAMBDA_COLLISION * collision_loss
             + LAMBDA_ENTROPY * entropy_loss)

    return total, nll_loss, seq_mask.sum()


def train(fpv_dir, raw_dir, variant, save_path, allowed_folders, epochs=50, batch_size=16,
          seed=None):
    assert variant in ('goal_conditioned', 'goal_zeroed')

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"fixed seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = FPVSequenceDataset(fpv_dir, raw_dir, allowed_folders=allowed_folders)
    if len(dataset) == 0:
        raise SystemExit(f"no sequences found under {fpv_dir} for scenes {allowed_folders}")
    print(f"{len(dataset)} ego sequences "
          f"({sum(len(s) for s in dataset.sequences)} supervised timesteps)")

    train_size = int(TRAIN_FRACTION * len(dataset))
    split_generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=split_generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_seq_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_seq_fn)

    model = StochasticFPVNavigationActor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"training variant={variant} for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        total_loss, total_steps = 0.0, 0.0
        start_time = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            loss, nll_loss, n_steps = batch_loss(model, batch, variant, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += nll_loss.item() * n_steps.item()
            total_steps += n_steps.item()

        model.eval()
        val_loss, val_steps = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, nll_loss, n_steps = batch_loss(model, batch, variant, device)
                val_loss += nll_loss.item() * n_steps.item()
                val_steps += n_steps.item()

        print(f"epoch {epoch + 1:03d}/{epochs} | train NLL {total_loss / total_steps:.4f} "
              f"| val NLL {val_loss / max(val_steps, 1):.4f} | {time.time() - start_time:.1f}s",
              flush=True)

        # Rolling checkpoint so a crashed run does not lose the whole split.
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path + ".ckpt")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    if os.path.exists(save_path + ".ckpt"):
        os.remove(save_path + ".ckpt")
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fpv-dir', required=True,
                    help='one FOV directory from build_fpv_dataset.py, e.g. <root>/135_deg')
    ap.add_argument('--raw-dir', required=True, help='matching raw ETH/UCY .txt directory')
    ap.add_argument('--variant', choices=['goal_conditioned', 'goal_zeroed'],
                    default='goal_conditioned')
    ap.add_argument('--out-dir', default='checkpoints')
    ap.add_argument('--split', choices=sorted(SPLIT_SCENES), default=None,
                    help='train only the model that holds this split out (default: all five)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=None,
                    help='seed torch/numpy/random and the train/val split (default: unseeded)')
    args = ap.parse_args()

    splits = [args.split] if args.split else sorted(SPLIT_SCENES)
    os.makedirs(args.out_dir, exist_ok=True)

    for held_out in splits:
        train_folders = [f for name, folders in SPLIT_SCENES.items() if name != held_out
                         for f in folders]
        save_path = os.path.join(args.out_dir, f"{args.variant}_{held_out}.pth")

        print(f"\n{'=' * 60}")
        print(f"leave-one-out: holding out {held_out}, training on {train_folders}")
        print(f"{'=' * 60}", flush=True)

        train(args.fpv_dir, args.raw_dir, args.variant, save_path, train_folders,
              epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)


if __name__ == '__main__':
    main()

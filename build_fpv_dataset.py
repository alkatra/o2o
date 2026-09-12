"""Stage 1: overhead -> onboard dataset construction.

Reads raw ETH/UCY annotation files (``<frame_id> <ped_id> <x> <y>``, tab- or space-separated)
and re-exports each scene once per pedestrian as that pedestrian's own egocentric observation
sequence.

For every annotated pedestrian at every timestep, every other agent present in that frame is
classified by casting 360 rays from the ego position against body ellipses (semi-axes 0.15 m
across x 0.25 m along the body axis, oriented by each agent's heading) out to a 10 m range:

  * an agent that owns the nearest ellipse intersection along at least one ray whose bearing
    falls inside the forward FOV cone is TRACKED (``green`` once it has been held for 5
    consecutive frames, ``yellow`` before that);
  * an agent visible to a ray outside the cone goes to the coarse blind-spot channel
    (``gray`` / ``black`` on the same 5-frame rule);
  * an agent no ray reaches first -- occluded by another body, or beyond the 10 m ray range --
    is ``red`` and is not observed at all this frame.

Output: ``<out_root>/<fov>_deg/<scene>/ego_<ped_id>.json``, one file per pedestrian per FOV
setting, consumed by ``fpv_dataset.FPVSequenceDataset``.

Usage:
    python build_fpv_dataset.py --raw-dir data/eth-standard/raw/all_data \
                                --out-root fpv_datasets/eth-standard
    python build_fpv_dataset.py --raw-dir <dir> --out-root <dir> --fov 90 135 180
    python build_fpv_dataset.py --raw-dir <dir> --out-root <dir> --scene biwi_eth
"""
import argparse
import json
import math
import os

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience, not a requirement
    def tqdm(it, **kwargs):
        return it

# Body ellipse semi-axes used as the occluding/occludable footprint of a pedestrian.
BODY_SEMI_AXIS_ACROSS = 0.15
BODY_SEMI_AXIS_ALONG = 0.25
NUM_RAYS = 360
MAX_RAY_RANGE = 10.0
# Frames of continuous tracking before an agent counts as "established" rather than "new".
TRACK_ESTABLISHED_FRAMES = 5
# ETH/UCY annotations are at 25 fps with one annotated frame every 10 raw frames (0.4 s).
SOURCE_FPS = 25.0
TIMESTEP_SECONDS = 0.4


class SceneData:
    """Raw ETH/UCY scene: per-pedestrian tracks and per-frame occupancy."""

    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.pedestrians = {}        # ped_id -> [(frame_id, x, y), ...] sorted by frame
        self.positions_by_frame = {}  # frame_id -> [(ped_id, x, y), ...]
        self.frames = []
        self._load()

    def _load(self):
        with open(self.filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                frame_id, ped_id = float(parts[0]), float(parts[1])
                x, y = float(parts[2]), float(parts[3])
                self.pedestrians.setdefault(ped_id, []).append((frame_id, x, y))
                self.positions_by_frame.setdefault(frame_id, []).append((ped_id, x, y))
        self.frames = sorted(self.positions_by_frame.keys())
        for ped_id in self.pedestrians:
            self.pedestrians[ped_id].sort(key=lambda p: p[0])


def get_agent_orientation_vector(trajectory, current_idx):
    """Finite-difference velocity at ``current_idx``; falls forward at the first sample."""
    if current_idx > 0:
        dx = trajectory[current_idx][1] - trajectory[current_idx - 1][1]
        dy = trajectory[current_idx][2] - trajectory[current_idx - 1][2]
        dt = (trajectory[current_idx][0] - trajectory[current_idx - 1][0]) / SOURCE_FPS
        if dt > 1e-4:
            return dx / dt, dy / dt
    elif current_idx + 1 < len(trajectory):
        dx = trajectory[current_idx + 1][1] - trajectory[current_idx][1]
        dy = trajectory[current_idx + 1][2] - trajectory[current_idx][2]
        dt = (trajectory[current_idx + 1][0] - trajectory[current_idx][0]) / SOURCE_FPS
        if dt > 1e-4:
            return dx / dt, dy / dt
    return 0.0, 0.0


def get_speed(trajectory, current_idx):
    if current_idx > 0:
        dx = trajectory[current_idx][1] - trajectory[current_idx - 1][1]
        dy = trajectory[current_idx][2] - trajectory[current_idx - 1][2]
        return math.hypot(dx, dy) / TIMESTEP_SECONDS
    return 0.0


def export_scene(scene_name, filepath, fovs=(90.0, 135.0, 180.0), out_root="fpv_datasets"):
    scene = SceneData(scene_name, filepath)

    theta = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)
    ray_dx, ray_dy = np.cos(theta), np.sin(theta)
    a, b = BODY_SEMI_AXIS_ACROSS, BODY_SEMI_AXIS_ALONG

    for ped_id, trajectory in tqdm(scene.pedestrians.items(), desc=f"peds in {scene_name}", leave=False):
        if len(trajectory) < 2:
            continue

        ego_frames = {p[0]: p for p in trajectory}
        start_frame = trajectory[0][0]
        states = {fov: {'hits': {}, 'output': []} for fov in fovs}

        for frame_id in sorted(ego_frames.keys()):
            ego_p = ego_frames[frame_id]
            origin_x, origin_y = ego_p[1], ego_p[2]

            idx = trajectory.index(ego_p)
            ego_vx, ego_vy = get_agent_orientation_vector(trajectory, idx)
            ego_angle = math.atan2(ego_vy, ego_vx)
            ego_speed = get_speed(trajectory, idx)

            active_peds = scene.positions_by_frame.get(frame_id, [])

            # --- ray/ellipse intersection: which agent each ray reaches first (FOV-independent) ---
            min_t = np.full(NUM_RAYS, MAX_RAY_RANGE)
            hit_agents = np.full(NUM_RAYS, -1.0, dtype=float)

            for other_id, cx, cy in active_peds:
                if other_id == ped_id:
                    continue

                other_traj = scene.pedestrians[other_id]
                o_idx = [p[0] for p in other_traj].index(frame_id)
                odx, ody = get_agent_orientation_vector(other_traj, o_idx)
                alpha = math.atan2(ody, odx)

                # Ego position and ray directions expressed in the other agent's body frame,
                # so the ellipse is axis-aligned and the intersection is a scalar quadratic.
                vx, vy = origin_x - cx, origin_y - cy
                cos_a, sin_a = np.cos(-alpha), np.sin(-alpha)

                v_prime_x = vx * cos_a - vy * sin_a
                v_prime_y = vx * sin_a + vy * cos_a
                d_prime_x = ray_dx * cos_a - ray_dy * sin_a
                d_prime_y = ray_dx * sin_a + ray_dy * cos_a

                qa = (d_prime_x / a) ** 2 + (d_prime_y / b) ** 2
                qb = 2 * ((v_prime_x * d_prime_x) / (a ** 2) + (v_prime_y * d_prime_y) / (b ** 2))
                qc = (v_prime_x / a) ** 2 + (v_prime_y / b) ** 2 - 1

                disc = qb ** 2 - 4 * qa * qc
                valid = disc >= 0
                if not np.any(valid):
                    continue

                sqrt_d = np.zeros_like(disc)
                sqrt_d[valid] = np.sqrt(disc[valid])

                t1 = np.full_like(disc, np.inf)
                t2 = np.full_like(disc, np.inf)
                t1[valid] = (-qb[valid] - sqrt_d[valid]) / (2 * qa[valid])
                t2[valid] = (-qb[valid] + sqrt_d[valid]) / (2 * qa[valid])
                t1[t1 < 1e-5] = np.inf
                t2[t2 < 1e-5] = np.inf
                t = np.minimum(t1, t2)

                closer = t < min_t
                min_t[closer] = t[closer]
                hit_agents[closer] = float(other_id)

            # --- FOV-dependent classification ---
            for fov in fovs:
                half_fov_rad = math.radians(fov) / 2.0
                hit_front, hit_back = set(), set()

                for i in range(NUM_RAYS):
                    hit_ped = hit_agents[i]
                    if hit_ped == -1.0:
                        continue
                    delta = abs(ego_angle - theta[i]) % (2 * math.pi)
                    if delta > math.pi:
                        delta = 2 * math.pi - delta
                    (hit_front if delta <= half_fov_rad else hit_back).add(hit_ped)

                timestep_data = {
                    "frame_id": frame_id,
                    "ego_agent": {
                        "ped_id": ped_id,
                        "egostart": [trajectory[0][1], trajectory[0][2]],
                        "egogoal": [trajectory[-1][1], trajectory[-1][2]],
                        "egocurrentspeed": ego_speed,
                        "egocurrentorientation": [ego_vx, ego_vy],
                        "egofullhistory": trajectory,
                    },
                    "other_agents": [],
                }

                consecutive_hits = states[fov]['hits']

                for other_id, cx, cy in active_peds:
                    if other_id == ped_id:
                        continue

                    f_ped = float(other_id)
                    is_front = f_ped in hit_front
                    is_back = f_ped in hit_back and not is_front

                    hit_count = consecutive_hits.get(other_id, 0)
                    if is_front or is_back:
                        hit_count += 1
                        if frame_id == start_frame:
                            hit_count = TRACK_ESTABLISHED_FRAMES
                    else:
                        hit_count = 0
                    consecutive_hits[other_id] = hit_count

                    if is_front:
                        color = 'green' if hit_count >= TRACK_ESTABLISHED_FRAMES else 'yellow'
                    elif is_back:
                        color = 'gray' if hit_count >= TRACK_ESTABLISHED_FRAMES else 'black'
                    else:
                        color = 'red'

                    other_traj = scene.pedestrians[other_id]
                    o_idx = [p[0] for p in other_traj].index(frame_id)
                    o_dx, o_dy = get_agent_orientation_vector(other_traj, o_idx)
                    o_speed = get_speed(other_traj, o_idx)
                    past_5 = [p for p in other_traj if p[0] <= frame_id][-5:]

                    timestep_data["other_agents"].append({
                        "ped_id": other_id,
                        "orientation": [o_dx, o_dy],
                        "speed": o_speed,
                        "color": color,
                        "tracked_timesteps": hit_count,
                        "past_5_history": past_5,
                    })

                states[fov]['output'].append(timestep_data)

        for fov in fovs:
            out_dir = os.path.join(out_root, f"{int(fov)}_deg", scene_name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"ego_{int(ped_id)}.json"), 'w') as f:
                json.dump(states[fov]['output'], f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-dir', required=True,
                    help='directory of raw ETH/UCY scene .txt files')
    ap.add_argument('--out-root', required=True,
                    help='output root; per-FOV subdirectories are created under it')
    ap.add_argument('--fov', type=float, nargs='+', default=[90.0, 135.0, 180.0],
                    help='forward FOV cone width(s) in degrees (default: 90 135 180)')
    ap.add_argument('--scene', nargs='*', default=None,
                    help='scene name(s) to export; default is every .txt in --raw-dir')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.raw_dir) if f.endswith('.txt'))
    if args.scene:
        wanted = set(args.scene)
        files = [f for f in files if f[:-4] in wanted]
    if not files:
        raise SystemExit(f"no matching .txt scene files in {args.raw_dir}")

    for filename in tqdm(files, desc="scenes"):
        scene_name = filename[:-4]
        export_scene(scene_name, os.path.join(args.raw_dir, filename),
                     fovs=tuple(args.fov), out_root=args.out_root)
        print(f"exported {scene_name} -> {args.out_root}/<fov>_deg/{scene_name}/")


if __name__ == '__main__':
    main()

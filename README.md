# Overhead to Onboard (O2O)

Code for *Overhead to Onboard: Adapting Overhead Crowd Datasets for Egocentric Robot
Navigation*: building the egocentric dataset, training the policy, and evaluating ADE/FDE.
The real-world robot deployment is not part of this release.

## Files

| File | Stage | What it does |
| --- | --- | --- |
| `build_eth_canonical.py` | 0 | Rebuilds the ETH-Canonical tree from the original `obsmat.txt` |
| `build_fpv_dataset.py` | 1 | Raw `.txt` → per-pedestrian egocentric JSON sequences |
| `fpv_dataset.py` | 1/2 | Torch `Dataset` over those sequences |
| `transforms.py` | all | Ego-frame transforms and the 16-sector blind-spot LiDAR |
| `model.py` | 2/3 | `PointNetFOV` crowd encoder + the policy |
| `train_loo.py` | 2 | Leave-one-out training, both variants |
| `eth_ucy_windows.py` | 3 | obs-8/pred-12 window builder |
| `evaluate.py` | 3 | Self-play rollout → ADE/FDE |
| `checkpoints/` | n/a | The ten trained policies behind the reported tables |

## Install

```bash
pip install -r requirements.txt
```

`torch` and `numpy` are required; `tqdm` is optional. `evaluate.py --variant orca` additionally
needs `rvo2` (Python-RVO2, not on PyPI); nothing else does.

## Data

Not redistributed. Bring the public ETH/UCY benchmark release. Two trees are used:

* `<data-root>/raw/all_data/`: one `.txt` per scene, input to stage 1
* `<data-root>/<split>/{train,val,test}/`: the five leave-one-out splits, used by stage 3

Lines are `<frame_id> <ped_id> <x> <y>`, one sample every 10 raw frames (0.4 s).

**ETH-Standard vs ETH-Canonical.** Standard is the community-redistributed file every
published baseline is trained on; its frame-rate convention inflates pedestrian speed by
roughly 1.7×. Canonical comes from the original release's `obsmat.txt`. The shipped
checkpoints and all reported comparisons use Standard, since that is what the baselines use.
To rebuild Canonical:

```bash
python build_eth_canonical.py --obsmat <ewap>/seq_eth/obsmat.txt \
    --standard-root <data-root-standard> --out-root <data-root-canonical>
```

Which pedestrian IDs belong to which split file is copied from the Standard tree; only
positions and timing come from `obsmat.txt`, so any difference you measure is the annotation,
not a reshuffled split. Frames are rescaled by 10/6, because `obsmat.txt` advances 6 raw units
per 0.4 s step while this pipeline assumes 10 and derives headings as `frame_delta / 25`.

## Stage 1: build the egocentric dataset

```bash
python build_fpv_dataset.py --raw-dir <data-root>/raw/all_data \
    --out-root fpv_datasets/std --fov 135
```

For each pedestrian at each timestep, 360 rays are cast against pedestrian body ellipses out
to 10 m. Agents inside the forward FOV cone with clear line of sight enter the visible set;
agents outside the cone enter the 16-sector blind-spot LiDAR; agents occluded or out of range
are unobserved. Writes one JSON per pedestrian, the per-agent expansion that multiplies each
scene by its agent count. `--fov 90 135 180` exports all three variants in one pass.

## Stage 2: train

```bash
python train_loo.py --fpv-dir fpv_datasets/std/135_deg --raw-dir <data-root>/raw/all_data \
    --variant goal_conditioned --out-dir checkpoints
python train_loo.py --fpv-dir fpv_datasets/std/135_deg --raw-dir <data-root>/raw/all_data \
    --variant goal_zeroed --out-dir checkpoints
```

Both variants share the network and input dimensionality, differing only in the goal channels:
the real local goal with `goal_valid = 1`, versus `(0, 0)` with `goal_valid = 0`. The flag is
load-bearing, because `(0, 0)` is already a genuine "arrived" goal vector in 4-9% of training
timesteps. Defaults: 50 epochs, batch 16, Adam 1e-3. Loss is Gaussian NLL of the demonstrated
next-step velocity plus speed-range (×1.5), goal-heading (×1.0), time-to-closest-approach
collision (×0.05) and entropy (×0.05) terms, all evaluated on the distribution mean.
`--split` trains one split; `--seed` fixes RNG and the train/val split.

## Stage 3: evaluate

```bash
python evaluate.py --data-root <data-root> --checkpoint-dir checkpoints \
    --variant goal_conditioned          # or goal_zeroed, or orca
```

Each held-out split is evaluated with the model that never trained on it. Every window is
rolled out in self-play: each agent runs its own copy of the policy with its own hidden state,
FOV/LiDAR state and goal, and all step synchronously, so each reacts to the others'
model-produced positions. Goals are the end of each agent's recorded track, looked up from its
last observed position. K=20 rollouts (rollout 0 is the deterministic mean); the winner is
chosen per window by displacement summed over all agents and timesteps, then ADE/FDE are
averaged over agent instances.

The K=1 columns are deterministic and reproduce the reported values exactly, as does ORCA. The
K=20 columns depend on the sampling RNG and agree to within ~1%, up to ~4% on ETH, which has
the fewest windows. `--seed` is applied per window, so results do not depend on window order.

## Checkpoints

Ten policies (5 splits × 2 variants), ETH-Standard at 135° FOV, named
`<variant>_<held_out_split>.pth`. Each trained on the other four splits and is only valid for
the split in its name; they load directly into `model.StochasticFPVNavigationActor`.

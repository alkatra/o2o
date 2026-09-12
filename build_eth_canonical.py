"""Reconstructs the ETH-Canonical split tree from the original release's ``obsmat.txt``.

The community-redistributed ETH file that every published baseline on this benchmark is
trained and evaluated on carries an uncorrected frame-rate convention, which inflates
pedestrian speed by roughly 1.7x relative to the original annotation. This script rebuilds
every ETH-derived file of that tree from the original annotation instead, so the two versions
can be compared directly.

``obsmat.txt`` columns: ``frame, ped_id, pos_x, pos_z, pos_y, v_x, v_z, v_y`` (the ``z``
columns are the vertical axis and are unused here). Output matches the benchmark convention:
``<frame>\\t<ped_id>\\t<x>\\t<y>``.

**Split membership is preserved exactly.** For each target file, the pedestrian IDs present in
the Standard version of that file are the IDs written to the Canonical version -- only
positions and timing come from the original annotation, never which pedestrian belongs to
which split. Any difference measured downstream is therefore attributable to the annotation
itself rather than to a reshuffled split.

**Frame rescaling.** ``obsmat.txt`` advances 6 raw units per 0.4 s annotation step, while this
pipeline (and the Standard file) assumes 10 raw units per step and derives headings as
``frame_delta / 25``. Left unscaled, a 6-unit gap would be read as 0.24 s instead of 0.4 s and
every derived heading and velocity would be inflated by ~1.67x. Every frame value is therefore
multiplied by exactly 10/6. Being a single uniform function, this preserves real-time ratios
for every gap -- not just the modal one -- and preserves same-instant co-presence exactly: two
pedestrians sharing a raw frame value before scaling still share one after. Positions are
never touched.

Usage:
    python build_eth_canonical.py --obsmat <ewap>/seq_eth/obsmat.txt \\
                                  --standard-root <data-root-standard> \\
                                  --out-root      <data-root-canonical>
"""
import argparse
import os

import numpy as np

# obsmat advances 6 raw frame units per 0.4 s step; the pipeline convention is 10.
FRAME_RESCALE = 10.0 / 6.0

# Every file in the split tree that is derived from the ETH scene.
DEFAULT_TARGETS = [
    "eth/test/biwi_eth.txt",
    "hotel/train/biwi_eth_train.txt",
    "hotel/val/biwi_eth_val.txt",
    "univ/train/biwi_eth_train.txt",
    "univ/val/biwi_eth_val.txt",
    "zara1/train/biwi_eth_train.txt",
    "zara1/val/biwi_eth_val.txt",
    "zara2/train/biwi_eth_train.txt",
    "zara2/val/biwi_eth_val.txt",
    "raw/all_data/biwi_eth.txt",
    "raw/train/biwi_eth_train.txt",
    "raw/val/biwi_eth_val.txt",
]


def load_obsmat(path):
    """Returns (frame_rescaled, ped_id, x, y) as one array, sorted later per target."""
    obsmat = np.loadtxt(path).copy()
    obsmat[:, 0] = obsmat[:, 0] * FRAME_RESCALE
    return obsmat


def rebuild(obsmat, standard_root, out_root, targets):
    ped_ids_all = obsmat[:, 1]
    n_written = 0

    for rel_path in targets:
        standard_path = os.path.join(standard_root, rel_path)
        out_path = os.path.join(out_root, rel_path)

        if not os.path.exists(standard_path):
            print(f"skipping {rel_path}: not present under {standard_root}")
            continue

        standard_data = np.loadtxt(standard_path)
        ped_ids = set(np.unique(standard_data[:, 1]).tolist())

        rows = obsmat[np.isin(ped_ids_all, list(ped_ids))]
        rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]  # by frame, then pedestrian

        missing = ped_ids - set(np.unique(rows[:, 1]).tolist())
        if missing:
            shown = sorted(missing)[:10]
            print(f"WARNING: {rel_path}: {len(missing)} pedestrian IDs present in the Standard "
                  f"file are absent from obsmat.txt: {shown}{'...' if len(missing) > 10 else ''}")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(f"{r[0]:.0f}\t{int(r[1])}\t{r[2]:.4f}\t{r[4]:.4f}\n")

        print(f"{rel_path}: {len(ped_ids)} peds requested, "
              f"{len(np.unique(rows[:, 1]))} written, {len(rows)} rows "
              f"(Standard file had {len(standard_data)} rows)")
        n_written += 1

    return n_written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--obsmat', required=True,
                    help="path to the original release's seq_eth/obsmat.txt")
    ap.add_argument('--standard-root', required=True,
                    help='root of the ETH-Standard split tree (read only; defines membership)')
    ap.add_argument('--out-root', required=True,
                    help='root to write the ETH-Canonical tree into')
    ap.add_argument('--targets', nargs='*', default=None,
                    help='override the list of ETH-derived files to rebuild')
    args = ap.parse_args()

    if os.path.abspath(args.standard_root) == os.path.abspath(args.out_root):
        raise SystemExit("--out-root must differ from --standard-root; "
                         "this would overwrite the Standard tree in place")

    obsmat = load_obsmat(args.obsmat)
    print(f"loaded {len(obsmat)} rows from {args.obsmat}, frames rescaled by 10/6")

    n = rebuild(obsmat, args.standard_root, args.out_root, args.targets or DEFAULT_TARGETS)
    print(f"\nrebuilt {n} files into {args.out_root}")
    print("non-ETH scenes are identical in both trees; copy or symlink them across if you want "
          "a complete tree.")


if __name__ == '__main__':
    main()

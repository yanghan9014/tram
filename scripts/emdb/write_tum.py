#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def invert_T(R_wc, t_wc):
    """Invert a world->cam pose to cam->world."""
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw

def to_tum_lines(Rs, ts, start=0.0, fps=None):
    """Yield TUM lines given cam->world Rs, ts."""
    n = len(Rs)
    if fps is None:  # integer timestamps if fps not provided
        stamps = np.arange(n, dtype=float)
    else:
        stamps = start + np.arange(n, dtype=float) / float(fps)

    for i in range(n):
        # TUM wants qx qy qz qw (unit quaternion)
        quat_xyzw = R.from_matrix(Rs[i]).as_quat()  # [x,y,z,w]
        # normalize for safety
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
        tx, ty, tz = ts[i].tolist()
        qx, qy, qz, qw = quat_xyzw.tolist()
        yield f"{stamps[i]:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"

def convert_npz_to_tum(npz_path: Path, out_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    Rs = np.asarray(data["pred_cam_R"])   # (N,3,3), cam->world
    Ts = np.asarray(data["pred_cam_T"])   # (N,3)

    lines = []
    for i in range(len(Rs)):
        qx, qy, qz, qw = R.from_matrix(Rs[i]).as_quat()  # xyzw
        tx, ty, tz = Ts[i].tolist()
        lines.append(f"{float(i):.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} poses to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert NPZ (pred_cam_R/T) to TUM RGB-D pose format.")
    ap.add_argument("npz", type=Path, help="Input .npz file")
    ap.add_argument("out", type=Path, help="Output poses.txt")
    args = ap.parse_args()
    convert_npz_to_tum(args.npz, args.out)

if __name__ == "__main__":
    main()

# def main():
#     ap = argparse.ArgumentParser(description="Convert NPZ poses to TUM RGB-D pose format.")
#     ap.add_argument("npz", type=Path, help="Input .npz file")
#     ap.add_argument("out", type=Path, help="Output poses.txt")
#     ap.add_argument("--which", choices=["pred_cam", "world_cam"], default="pred_cam",
#                     help="Choose which pose stream to use (prefix for _R, _T)")
#     ap.add_argument("--assume", choices=["cam2world", "world2cam"], default="cam2world",
#                     help="Assumed direction of the chosen matrices before optional invert.")
#     ap.add_argument("--to", choices=["cam2world", "world2cam"], default="cam2world",
#                     help="Direction to output (TUM expects camera pose in world = cam2world).")
#     ap.add_argument("--fps", type=float, default=None, help="If set, timestamps are i/fps; else 0..N-1")
#     ap.add_argument("--start", type=float, default=0.0, help="Start time (seconds) if --fps is used")
#     args = ap.parse_args()

#     data = np.load(args.npz, allow_pickle=False)
#     R_key = f"{args.which}_R"
#     T_key = f"{args.which}_T"
#     if R_key not in data or T_key not in data:
#         raise KeyError(f"Expected keys '{R_key}' and '{T_key}' in {args.npz.name} (found {list(data.keys())})")

#     Rs = np.asarray(data[R_key])  # (N,3,3)
#     Ts = np.asarray(data[T_key])  # (N,3)
#     assert Rs.ndim == 3 and Rs.shape[1:] == (3,3)
#     assert Ts.ndim == 2 and Ts.shape[1] == 3
#     N = Rs.shape[0]
#     if Ts.shape[0] != N:
#         raise ValueError(f"Length mismatch: {R_key} has {N}, {T_key} has {Ts.shape[0]}")

#     # Decide whether to invert to get cam->world for output
#     need_invert = (args.assume != args.to)

#     Rs_out = np.empty_like(Rs)
#     Ts_out = np.empty_like(Ts)
#     if need_invert:
#         for i in range(N):
#             Rs_out[i], Ts_out[i] = invert_T(Rs[i], Ts[i])
#     else:
#         Rs_out, Ts_out = Rs, Ts

#     # If user asked for world2cam output, warn (TUM expects cam->world)
#     if args.to != "cam2world":
#         print("Note: TUM format usually expects camera pose in world (cam2world). You chose world2cam.")

#     lines = list(to_tum_lines(Rs_out, Ts_out, start=args.start, fps=args.fps))
#     args.out.write_text("\n".join(lines) + "\n")
#     print(f"Wrote {len(lines)} poses to {args.out}")

# if __name__ == "__main__":
#     main()

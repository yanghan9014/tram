#!/usr/bin/env python3
import argparse, subprocess, sys, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Save the left half of an MP4.")
    ap.add_argument("input_mp4", help="path to input .mp4")
    ap.add_argument("output_mp4", nargs="?", help="path to output .mp4 (default: <input>_left.mp4)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Please install it and try again.")

    in_path = Path(args.input_mp4)
    out_path = Path(args.output_mp4) if args.output_mp4 else in_path.with_name(in_path.stem + "_left.mp4")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-filter:v", "crop=iw/2:ih:0:0",
        "-c:a", "copy",
        str(out_path)
    ], check=True)

if __name__ == "__main__":
    main()

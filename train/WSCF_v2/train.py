"""Launch YOLOv12-WSCF v2's explicit two-stage training from the repository root."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="YOLOv12-WSCF v2 two-stage training")
    parser.add_argument("stage", choices=("a", "b"), help="a: initialize added modules; b: full fine-tuning")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    script = Path(__file__).resolve().parent / (
        "train_stageA.py" if args.stage == "a" else "train_stageB.py"
    )
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

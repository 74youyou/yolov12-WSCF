"""Compatibility entry for YOLOv12-WSCF V1 stage B."""
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    print("Legacy path: running WSCF V1 stage B from train/WSCF_v1.")
    runpy.run_path(str(root / "train" / "WSCF_v1" / "train_stageB.py"), run_name="__main__")


if __name__ == "__main__":
    main()

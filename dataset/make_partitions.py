"""Generate the client partitions used in the paper from the raw data.

Usage (from the dataset/ folder, raw files in <dataset>/rawdata/):
    python make_partitions.py nslkdd unsw cic_ids toniot

For every dataset this writes
    <dataset>/noniid20/       pat partition (2 classes per client)
    <dataset>/noniidexdir20/  exDir partition (Dirichlet, alpha = 0.1)
    <dataset>/centralized/    pooled data for the Centralized reference
which are the folder names read by system/main.py (-dt noniid / -dt exdir).

Each partition is generated in a fresh Python process because the generate_<dataset>.py
modules seed random/numpy (seed 1) at import time; generating two partitions in one
process would change the random state of the second one.
"""
import shutil
import subprocess
import sys
from pathlib import Path

# folder written by generate_dataset(...) -> folder read by system/main.py
OUT_NAMES = {
    "pat": ["noniid20pat", "noniid20pat01"],
    "exdir": ["noniid20exdir", "noniid20exdir01"],
}
TARGET = {"pat": "noniid20", "exdir": "noniidexdir20"}


def run(code):
    subprocess.run([sys.executable, "-c", code], check=True)


def main(datasets):
    for d in datasets:
        mod = f"generate_{d}"
        for part in ["pat", "exdir"]:
            target = Path(d) / TARGET[part]
            if target.exists():
                print(f"{target} exists, skipped")
                continue
            # niid=True, balance=False (unbalanced client sizes), partition = pat / exdir
            run(f"import {mod} as g; g.generate_dataset(g.dir_path, g.num_clients, True, False, '{part}')")
            for name in OUT_NAMES[part]:
                if (Path(d) / name).exists():
                    shutil.move(str(Path(d) / name), str(target))
                    break
            print(f"{target} done")
        if not (Path(d) / "centralized").exists():
            run(f"import {mod} as g; g.generate_centralized_dataset(g.dir_path)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nslkdd", "unsw", "cic_ids", "toniot"])

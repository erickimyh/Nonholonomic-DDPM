"""
Grid sweep over run.py hyperparameters.

Edit GRID / FIXED below, then:

    python sweep.py                  # sequential
    python sweep.py --workers 4      # 4 runs at a time, 1 CPU thread each
    python sweep.py --dry-run        # print the commands without running them

Each run is a separate `python run.py ...` subprocess, so W&B, logging and
torch state never leak between runs. Boolean flags (e.g. wandb-offline) are
passed as bare switches when their value is True and omitted when False.
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Sweep definition ──────────────────────────────────────────────────────────

# Every combination of these values is run.
GRID = {
    'schedule': ['exponential'],
    'loss-weight': ['time-weighted'],
    't0': [1e-4, 1e-3],
}

# Passed unchanged to every run.
FIXED = {
    'loss-type': 'ibp',
    'epochs': 500,
    'N':      5000,
    'eval-every': 50, 
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_runs(prefix: str) -> list[tuple[str, list[str]]]:
    """Return (run_name, command) for every point in GRID."""
    keys = list(GRID)
    runs = []
    for values in itertools.product(*GRID.values()):
        point = dict(zip(keys, values))
        run_name = prefix + '_'.join(f'{k}={v}' for k, v in point.items())
        cmd = [sys.executable, 'run.py', '--run-name', run_name]
        for k, v in {**FIXED, **point}.items():
            if isinstance(v, bool):
                if v:
                    cmd.append(f'--{k}')
            else:
                cmd += [f'--{k}', str(v)]
        runs.append((run_name, cmd))
    return runs


def run_one(run_name: str, cmd: list[str], parallel: bool, log_dir: Path) -> tuple[str, int, float]:
    """Run one subprocess; returns (run_name, return code, minutes)."""
    t_start = time.time()
    if parallel:
        # Interleaved stdout from several runs is unreadable, so each run's
        # console output goes to its own file. One thread per process stops
        # the workers from fighting over cores.
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with open(log_dir / f'{run_name}.log', 'w') as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run(cmd).returncode
    return run_name, rc, (time.time() - t_start) / 60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep over run.py.")
    p.add_argument('--workers', type=int, default=1,
                   help="Number of runs executed at once (1 = sequential).")
    p.add_argument('--prefix', type=str, default='',
                   help="String prepended to every run name, e.g. 'sweep1_'.")
    p.add_argument('--dry-run', action='store_true',
                   help="Print the commands and exit.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    runs = build_runs(args.prefix)
    print(f"{len(runs)} runs, {args.workers} worker(s)")

    if args.dry_run:
        for _, cmd in runs:
            print(' '.join(cmd))
        return

    parallel = args.workers > 1
    log_dir = Path('output') / 'sweep_logs'
    if parallel:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Console output of each run -> {log_dir}/<run-name>.log")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, name, cmd, parallel, log_dir) for name, cmd in runs]
        for i, fut in enumerate(as_completed(futures), 1):
            name, rc, minutes = fut.result()
            status = 'ok' if rc == 0 else f'FAILED (exit {rc})'
            print(f"[{i}/{len(runs)}] {name}: {status}, {minutes:.1f} min")
            if rc != 0:
                failed.append(name)

    if failed:
        print(f"\n{len(failed)} run(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll runs finished.")


if __name__ == '__main__':
    main()

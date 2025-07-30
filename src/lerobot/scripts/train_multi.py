#!/usr/bin/env python

import subprocess
import os
import sys
import time

"""
Multi-agent launcher for lerobot training.

This script launches 6 training processes:
- 3 on GPU 0
- 3 on GPU 1

Each process runs train.py with the appropriate device assignment.

Usage:
    python train_multi.py [additional train.py args]

All additional arguments are passed to each train.py process.
"""

NUM_GPUS = 2
AGENTS_PER_GPU = 1
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")

def launch_agent(gpu_id, agent_idx, extra_args):
    # Set CUDA_VISIBLE_DEVICES to restrict to one GPU per process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Each agent can have a different seed if desired
    seed = 1000 + gpu_id * 10 + agent_idx
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        f"--policy.device=cuda",  # Pass as single argument with '='
        f"--seed={seed}",
    ] + extra_args
    print(f"Launching agent {agent_idx+1} on GPU {gpu_id} with seed {seed}")
    return subprocess.Popen(cmd, env=env)

def main():
    extra_args = sys.argv[1:]
    procs = []
    for gpu_id in range(NUM_GPUS):
        for agent_idx in range(AGENTS_PER_GPU):
            proc = launch_agent(gpu_id, agent_idx, extra_args)
            procs.append(proc)
            time.sleep(1)  # Stagger launches slightly

    # Wait for all processes to finish
    for proc in procs:
        proc.wait()

if __name__ == "__main__":
    main()
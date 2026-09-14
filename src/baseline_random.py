import os
import sys
import random
from datetime import datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.home() / "alpasim"
BASE_RUNS_DIR = PROJECT_ROOT / "autolab_baselines" / "random"
TARGET_ITERATIONS = 50
DELAY_CHOICES = [50000, 75000, 100000, 125000, 150000, 175000, 200000]

BASE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

for i in range(1, TARGET_ITERATIONS + 1):
    offset = round(random.uniform(3.5, 5.5), 2)
    delay = random.choice(DELAY_CHOICES)
    run_dir = BASE_RUNS_DIR / f"run_{i:03d}_{datetime.now().strftime('%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "alpasim_wizard",
        "deploy=local", "topology=1gpu", "driver=vavam",
        f"wizard.log_dir={run_dir}",
        "runtime.simulation_config.force_gt_duration_us=2500000",
        f"runtime.simulation_config.route_start_offset_m={offset}",
        f"runtime.simulation_config.planner_delay_us={delay}",
    ]

    print(f"\n[RANDOM {i:03d}/{TARGET_ITERATIONS:03d}] offset={offset}m, delay={delay}us")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/src/wizard:{env.get('PYTHONPATH', '')}"

    import subprocess
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(run_dir / "console.log", "w") as f:
        f.write(proc.stdout)

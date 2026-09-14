import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
from lora_agent import query_agent

PROJECT_ROOT = Path.home() / "alpasim"
BASE_RUNS_DIR = PROJECT_ROOT / "autolab_experiments"
TARGET_ITERATIONS = 50


def parse_telemetry(run_dir: Path) -> dict:
    """Extract safety and driving quality metrics from AlpaSim output artifacts."""
    metrics = {
        "collision_at_fault": False,
        "collision_any": False,
        "dist_to_gt_trajectory": 0.0,
        "progress_rel": 0.0,
        "offroad": False,
        "open_loop_collision": 0.0,
        "plan_deviation": 0.0,
        "status": "COMPLETED"
    }

    parquet_candidates = [
        run_dir / "aggregate" / "metrics_results.parquet",
        run_dir / "aggregate" / "metrics_unprocessed.parquet",
    ]
    parquet_candidates.extend(list(run_dir.glob("rollouts/**/metrics.parquet")))

    found_parquet = next((p for p in parquet_candidates if p.exists()), None)

    if found_parquet:
        try:
            df = pd.read_parquet(found_parquet)
            if not df.empty:
                last_row = df.iloc[-1].to_dict()
                metrics["collision_at_fault"] = bool(last_row.get("collision_at_fault", False))
                metrics["collision_any"] = bool(last_row.get("collision_any", False))
                metrics["dist_to_gt_trajectory"] = float(last_row.get("dist_to_gt_trajectory", 0.0))
                metrics["progress_rel"] = float(last_row.get("progress_rel", 0.0))
                metrics["offroad"] = bool(last_row.get("offroad", False))
                metrics["open_loop_collision"] = float(last_row.get("open_loop_collision", 0.0))
                metrics["plan_deviation"] = float(last_row.get("plan_deviation", 0.0))
                return metrics
        except Exception as e:
            print(f"[!] Error reading parquet metrics {found_parquet}: {e}")

    summary_json = run_dir / "results-summary.json"
    if summary_json.exists():
        try:
            with open(summary_json) as f:
                data = json.load(f)
                metrics["status"] = data.get("status", "COMPLETED")
                return metrics
        except Exception as e:
            print(f"[!] Error reading summary json: {e}")

    metrics["status"] = "MISSING_DATA"
    return metrics


def run_simulation(iteration: int, knobs: dict) -> dict:
    """Invoke alpasim_wizard with dynamic Hydra overrides."""
    run_dir = BASE_RUNS_DIR / f"run_{iteration:03d}_{datetime.now().strftime('%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m", "alpasim_wizard",
        "deploy=local",
        "topology=1gpu",
        "driver=vavam",
        f"wizard.log_dir={run_dir}",
        f"runtime.simulation_config.force_gt_duration_us={knobs['force_gt_duration_us']}",
        f"runtime.simulation_config.route_start_offset_m={knobs['route_start_offset_m']}",
        f"runtime.simulation_config.planner_delay_us={knobs['planner_delay_us']}",
    ]

    print(f"\n=================================================================")
    print(f" ITERATION {iteration:03d} / {TARGET_ITERATIONS:03d}")
    print(f" Knobs: offset={knobs['route_start_offset_m']}m | delay={knobs['planner_delay_us']}us | warmup={knobs['force_gt_duration_us']}us")
    print(f" Rationale: {knobs.get('reasoning')}")
    print(f" Log directory: {run_dir}")
    print(f"=================================================================")

    start_time = datetime.now()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/src/wizard:{env.get('PYTHONPATH', '')}"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )

        with open(run_dir / "console.log", "w") as f:
            f.write(proc.stdout)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[*] Simulation step completed in {elapsed:.1f}s (Exit code: {proc.returncode})")

        metrics = parse_telemetry(run_dir)
        if proc.returncode != 0 and metrics["status"] == "MISSING_DATA":
            metrics["status"] = "SIMULATION_CRASHED"

        return metrics

    except Exception as e:
        print(f"[!] Simulation execution failed: {e}")
        return {"status": "EXECUTION_ERROR", "error": str(e)}


def main():
    BASE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    current_knobs = {
        "force_gt_duration_us": 2500000,
        "route_start_offset_m": 4.5,
        "planner_delay_us": 50000,
        "reasoning": "Checkpoint 2 hardened baseline."
    }
    current_metrics = {
        "collision_at_fault": False,
        "collision_any": False,
        "open_loop_collision": 0.0,
        "plan_deviation": 0.0,
        "status": "INIT"
    }

    print(f"Starting Checkpoint 2 batch on CPU. Target iterations: {TARGET_ITERATIONS}")

    for i in range(1, TARGET_ITERATIONS + 1):
        outcome = run_simulation(i, current_knobs)

        print(
            f"Outcome [{i}]: Fault={outcome.get('collision_at_fault')} | "
            f"AnyCollision={outcome.get('collision_any')} | "
            f"OpenLoopColl={outcome.get('open_loop_collision')} | "
            f"PlanDev={outcome.get('plan_deviation'):.2f}m | "
            f"Progress={outcome.get('progress_rel', 0.0)*100:.1f}%"
        )

        if i < TARGET_ITERATIONS:
            current_knobs = query_agent(outcome, current_knobs)
            current_metrics = outcome


if __name__ == "__main__":
    main()

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:3b"

PROJECT_ROOT = Path.home() / "alpasim"
BASE_RUNS_DIR = PROJECT_ROOT / "autolab_experiments"
DATASET_LOG_FILE = PROJECT_ROOT / "autolab_finetune_dataset.jsonl"

SYSTEM_PROMPT = """You are an Autonomous Vehicle safety verification and red-teaming agent.
Your objective is to expose controller failures and near-collision edge cases in AlpaSim by mutating two parameters:
1. 'force_gt_duration_us': Ground-truth warmup duration in microseconds.
   - MUST be an exact multiple of 500000.
   - Allowed values: 500000, 1000000, 1500000, 2000000, 2500000.
2. 'route_start_offset_m': Longitudinal offset in meters where the ego vehicle enters the route.
   - Valid float range: [3.5, 5.5].

Analyze the previous run telemetry. If the run was safe (no collision, low trajectory deviation), push the parameters toward tighter reaction margins. If the run crashed or went off-road, explore the critical sensitivity boundary near those values.

You must respond ONLY with a single valid JSON object strictly matching this schema:
{
  "force_gt_duration_us": int,
  "route_start_offset_m": float,
  "reasoning": "brief justification of the chosen parameter shift"
}"""

# -----------------------------------------------------------------------------
# LLM Orchestration
# -----------------------------------------------------------------------------
def query_agent(last_metrics: dict, last_knobs: dict) -> dict:
    """Prompt the local Ollama instance to generate the next parameter mutation."""
    if last_metrics.get("status") == "SIMULATION_CRASHED":
        user_prompt = (
            f"Previous Parameters: {json.dumps(last_knobs)}\n"
            "Execution Status: SPAWN_INITIALIZATION_CRASH.\n"
            "The vehicle pose failed before simulation start.\n"
            "Constraint: Keep route_start_offset_m strictly between 3.5 and 5.5 meters.\n\n"
            "Generate the next parameter set to stress-test the controller."
        )
    else:
        user_prompt = (
            f"Previous Parameters: {json.dumps(last_knobs)}\n"
            f"Previous Execution Telemetry: {json.dumps(last_metrics)}\n\n"
            "Generate the next parameter set to stress-test the controller."
        )

    payload = {
        "model": MODEL_NAME,
        "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 256
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        raw_output = response.json()["response"]
        parsed = json.loads(raw_output)

        # Sanitize and clamp values to safe domain bounds
        # Round force_gt_duration_us to nearest multiple of 500,000 us (0.5s steps)
        raw_gt = parsed.get("force_gt_duration_us", 1500000)
        clamped_gt = max(500000, min(3000000, raw_gt))
        valid_gt = int(round(clamped_gt / 500000.0) * 500000)

        knobs = {
            "force_gt_duration_us": valid_gt,
            "route_start_offset_m": round(float(max(3.5, min(5.5, parsed.get("route_start_offset_m", 4.5)))), 2),
            "reasoning": str(parsed.get("reasoning", "No rationale provided."))
        }
        return knobs

    except Exception as e:
        print(f"[!] Warning: LLM query failed or produced malformed JSON ({e}). Using heuristic fallback.")
        return {
            "force_gt_duration_us": 1500000,
            "route_start_offset_m": 4.5,
            "reasoning": "Fallback mutation due to inference error."
        }

# -----------------------------------------------------------------------------
# Telemetry Parser
# -----------------------------------------------------------------------------
def parse_telemetry(run_dir: Path) -> dict:
    """Extract safety and driving quality metrics from AlpaSim output artifacts."""
    metrics = {
        "collision_at_fault": False,
        "collision_any": False,
        "dist_to_gt_trajectory": 0.0,
        "progress_rel": 0.0,
        "offroad": False,
        "status": "COMPLETED"
    }

    # 1. Check for aggregate metrics parquet (preferred)
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
                return metrics
        except Exception as e:
            print(f"[!] Error reading parquet metrics {found_parquet}: {e}")

    # 2. Check for results-summary.json fallback
    summary_json = run_dir / "results-summary.json"
    if summary_json.exists():
        try:
            with open(summary_json) as f:
                data = json.load(f)
                metrics["status"] = data.get("status", "COMPLETED")
                return metrics
        except Exception as e:
            print(f"[!] Error reading summary json: {e}")

    print("[!] Notice: No structured metrics artifact found. Check console logs in run directory.")
    metrics["status"] = "MISSING_DATA"
    return metrics

# -----------------------------------------------------------------------------
# Simulation Runner
# -----------------------------------------------------------------------------
def run_simulation(iteration: int, knobs: dict) -> dict:
    """Invoke alpasim_wizard with dynamic Hydra overrides."""
    run_dir = BASE_RUNS_DIR / f"run_{iteration:03d}_{datetime.now().strftime('%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PROJECT_ROOT / ".venv/bin/alpasim_wizard"),
        "deploy=local",
        "topology=1gpu",
        "driver=vavam",
        f"wizard.log_dir={run_dir}",
        f"runtime.simulation_config.force_gt_duration_us={knobs['force_gt_duration_us']}",
        f"runtime.simulation_config.route_start_offset_m={knobs['route_start_offset_m']}",
    ]

    print(f"\n=================================================================")
    print(f" ITERATION {iteration:03d}")
    print(f" Knobs: force_gt_duration_us={knobs['force_gt_duration_us']} | route_start_offset_m={knobs['route_start_offset_m']}")
    print(f" Rationale: {knobs.get('reasoning')}")
    print(f" Log directory: {run_dir}")
    print(f"=================================================================")

    start_time = datetime.now()
    try:
        # Run subprocess from the project root where the AlpaSim environment lives
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        with open(run_dir / "wizard_output.log", "w") as log_file:
            log_file.write(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[X] Simulation failed with returncode {e.returncode}")
        with open(run_dir / "crash_error.log", "w") as err_file:
            err_file.write(e.stdout if e.stdout else str(e))
        return {
            "collision_at_fault": False,
            "collision_any": False,
            "dist_to_gt_trajectory": 99.0,
            "progress_rel": 0.0,
            "offroad": True,
            "status": "SIMULATION_CRASHED"
        }
    finally:
        # Tear down containers and prune stale networks
        compose_file = run_dir / "docker-compose.yaml"
        if compose_file.exists():
            subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        subprocess.run(["docker", "network", "prune", "-f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[*] Simulation step completed in {elapsed:.1f}s")

    metrics = parse_telemetry(run_dir)
    return metrics

# -----------------------------------------------------------------------------
# Main Loop & Dataset Recorder
# -----------------------------------------------------------------------------
def log_dataset_sample(prompt_metrics: dict, chosen_knobs: dict, result_metrics: dict):
    """Save transitions for future QLoRA fine-tuning on the RTX 4070."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "input_state": prompt_metrics,
        "agent_decision": chosen_knobs,
        "simulation_outcome": result_metrics
    }
    with open(DATASET_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def main():
    total_iterations = 100
    BASE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Initial baseline seeds
    current_knobs = {
        "force_gt_duration_us": 1500000,
        "route_start_offset_m": 4.5,
        "reasoning": "Baseline standard execution"
    }
    current_metrics = {
        "collision_at_fault": False,
        "collision_any": False,
        "dist_to_gt_trajectory": 0.05,
        "progress_rel": 1.0,
        "offroad": False,
        "status": "INITIAL_SEED"
    }

    print(f"Starting AutoLab closed loop on device. Target iterations: {total_iterations}")

    for i in range(1, total_iterations + 1):
        # 1. Ask Qwen 2.5 3B for next mutation
        next_knobs = query_agent(current_metrics, current_knobs)

        # 2. Run simulation with chosen knobs
        new_metrics = run_simulation(i, next_knobs)

        # 3. Log data for Dr. Shao's fine-tuning phase
        log_dataset_sample(current_metrics, next_knobs, new_metrics)

        # 4. Advance states
        current_knobs = next_knobs
        current_metrics = new_metrics

        print(f"Outcome [{i}]: Fault={current_metrics['collision_at_fault']} | "
              f"AnyCollision={current_metrics['collision_any']} | "
              f"MaxDev={current_metrics['dist_to_gt_trajectory']:.2f}m | "
              f"Progress={current_metrics['progress_rel']*100:.1f}%")

    print("\nAutoLab batch run complete. Data saved to autolab_finetune_dataset.jsonl")

if __name__ == "__main__":
    main()
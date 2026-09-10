import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Force headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_PATH = Path.home() / "alpasim" / "autolab_dataset_clean.jsonl"
OUTPUT_PLOT = Path.home() / "alpasim" / "failure_boundary.png"

def load_data(filepath: Path) -> pd.DataFrame:
    records = []
    if not filepath.exists():
        filepath = Path.home() / "alpasim" / "autolab_finetune_dataset.jsonl"

    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("simulation_outcome", {}).get("status") != "COMPLETED":
                continue

            decision = entry.get("agent_decision", {})
            outcome = entry.get("simulation_outcome", {})
            records.append({
                "offset_m": decision.get("route_start_offset_m"),
                "duration_s": decision.get("force_gt_duration_us", 0) / 1e6,
                "collision_at_fault": outcome.get("collision_at_fault", False),
                "collision_any": outcome.get("collision_any", False),
                "deviation_m": outcome.get("dist_to_gt_trajectory", 0.0),
            })
    return pd.DataFrame(records)

def generate_plot(df: pd.DataFrame):
    plt.figure(figsize=(10, 6), dpi=300)

    # Apply deterministic jitter to reveal duplicate evaluations
    np.random.seed(42)
    df["offset_jitter"] = df["offset_m"] + np.random.uniform(-0.03, 0.03, size=len(df))
    df["duration_jitter"] = df["duration_s"] + np.random.uniform(-0.03, 0.03, size=len(df))

    faults = df[df["collision_at_fault"] == True]
    safe = df[df["collision_at_fault"] == False]

    plt.scatter(
        safe["offset_jitter"],
        safe["duration_jitter"],
        color="#2ecc71",
        label=f"Safe Run (n={len(safe)})",
        alpha=0.6,
        s=90,
        edgecolors="k",
        linewidths=0.5
    )
    plt.scatter(
        faults["offset_jitter"],
        faults["duration_jitter"],
        color="#e74c3c",
        marker="X",
        label=f"Collision Fault (n={len(faults)})",
        alpha=0.7,
        s=110,
        edgecolors="k",
        linewidths=0.5
    )

    plt.title("NMPC Controller Kinematic Sensitivity Boundary (Jittered)", fontsize=14, weight="bold")
    plt.xlabel("Route Start Offset (m)", fontsize=12)
    plt.ylabel("Ground-Truth Warmup Duration (s)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT)
    print(f"[*] Plot saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    if df.empty:
        print("[!] No completed rollout records found.")
    else:
        generate_plot(df)
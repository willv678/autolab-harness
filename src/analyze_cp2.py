from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

exp_dir = Path.home() / "alpasim" / "autolab_experiments"
runs = sorted(exp_dir.glob("run_*"))[-50:]

records = []
for r in runs:
    pqs = list(r.glob("**/metrics*.parquet"))
    if not pqs:
        continue
    try:
        df = pd.read_parquet(pqs[0])
        # Read console log to extract input knobs if not in parquet
        delay = 0
        offset = 4.5
        log_file = r / "console.log"
        if log_file.exists():
            with open(log_file) as f:
                content = f.read()
                for line in content.splitlines():
                    if "planner_delay_us=" in line:
                        delay = int(line.split("planner_delay_us=")[1].split()[0].strip(","))
                    if "route_start_offset_m=" in line:
                        offset = float(line.split("route_start_offset_m=")[1].split()[0].strip(","))

        records.append({
            "run": r.name,
            "collision": bool(df.get("collision_at_fault", [False])[0]),
            "open_loop": float(df.get("open_loop_collision", [0.0])[0]),
            "plan_dev": float(df.get("plan_deviation", [0.0])[0]),
            "progress": float(df.get("progress_rel", [0.0])[0]),
            "delay_ms": delay / 1000.0,
            "offset_m": offset,
        })
    except Exception:
        continue

res = pd.DataFrame(records)
print(f"Total Parsed Checkpoint 2 Runs: {len(res)}")
if len(res) == 0:
    exit()

crashes = res[res["collision"] == True]
safes = res[res["collision"] == False]

print(f"Collision Count: {len(crashes)} / {len(res)} ({len(crashes)/len(res):.1%})")
print(f"Mean Plan Deviation (Safe): {safes['plan_dev'].mean():.2f}m")
if len(crashes) > 0:
    print(f"Mean Plan Deviation (Crash): {crashes['plan_dev'].mean():.2f}m")
    print(f"Mean Open-Loop Collision Rate: {crashes['open_loop'].mean():.2f}")

# Plot boundary: Latency vs Offset
plt.figure(figsize=(8, 5))
plt.scatter(safes["offset_m"], safes["delay_ms"], c="mediumseagreen", label=f"Safe (n={len(safes)})", s=60, alpha=0.8)
if len(crashes) > 0:
    plt.scatter(crashes["offset_m"], crashes["delay_ms"], c="crimson", marker="x", label=f"Collision (n={len(crashes)})", s=80)

plt.title("VaVAM Planner Robustness Boundary (Checkpoint 2: Warmup = 2.5s)")
plt.xlabel("Route Start Offset (m)")
plt.ylabel("Injected Planner Latency (ms)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("cp2_failure_boundary.png", dpi=300)
print("[*] Plot saved to cp2_failure_boundary.png")

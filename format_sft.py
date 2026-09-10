import json
import random
from pathlib import Path

INPUT_FILE = Path.home() / "alpasim" / "autolab_dataset_clean.jsonl"
TRAIN_OUTPUT = Path.home() / "alpasim" / "sft_train.jsonl"
VAL_OUTPUT = Path.home() / "alpasim" / "sft_val.jsonl"

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

def convert_record(entry: dict) -> dict:
    input_state = entry["input_state"]
    decision = entry["agent_decision"]

    user_content = (
        f"Previous Execution Telemetry: {json.dumps(input_state)}\n\n"
        "Generate the next parameter set to stress-test the controller."
    )
    assistant_content = json.dumps(decision)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def main():
    if not INPUT_FILE.exists():
        print(f"[!] Input file not found: {INPUT_FILE}")
        return

    records = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("simulation_outcome", {}).get("status") == "COMPLETED":
                records.append(convert_record(item))

    print(f"[*] Total valid transitions extracted: {len(records)}")
    if not records:
        return

    random.seed(42)
    random.shuffle(records)

    split_idx = int(len(records) * 0.8)
    train_set = records[:split_idx]
    val_set = records[split_idx:]

    with open(TRAIN_OUTPUT, "w") as f:
        for r in train_set:
            f.write(json.dumps(r) + "\n")

    with open(VAL_OUTPUT, "w") as f:
        for r in val_set:
            f.write(json.dumps(r) + "\n")

    print(f"[*] Train set ({len(train_set)}) written to: {TRAIN_OUTPUT}")
    print(f"[*] Validation set ({len(val_set)}) written to: {VAL_OUTPUT}")

if __name__ == "__main__":
    main()
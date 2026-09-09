# AutoLab: LLM-Driven Autonomous Vehicle Red-Teaming Harness

An agentic test harness designed to systematically discover failure boundaries and edge cases in autonomous driving controllers using closed-loop simulation.

## Architecture
1. **Test-Master Agent:** Local reasoning LLM (e.g., Qwen 2.5) that analyzes prior telemetry and proposes mutated scenario parameters.
2. **Simulation Bridge:** Programmatically injects configuration overrides and triggers containerized multi-service validation.
3. **Telemetry Engine:** Ingests parquet metrics and event logs to evaluate trajectory deviation, progress, and fault conditions.

## Quick Start
```bash
python autolab_loop.py```

### 5. Push to GitHub

1. Go to GitHub and create a **new public repository** named `autolab-harness` (leave it completely empty—don't initialize with a README or license, since you just made your own).
2. Link and push your local repo:

```bash
git add .
git commit -m "feat: Initial commit of AutoLab agent harness and loop orchestration"
git branch -M main
git remote add origin https://github.com/<your-github-username>/autolab-harness.git
git push -u origin main

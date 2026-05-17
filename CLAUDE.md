# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrafficVision is a traffic-light control system for the NTUT (National Taipei University of Technology) campus area. It combines real-world VD (Vehicle Detection) sensor data, SUMO traffic simulation, a GRU neural network for short-horizon traffic prediction, and a signal-optimization layer. Outputs are served via a FastAPI dashboard and optionally explained by a fine-tuned Gemma 4 model running in Ollama.

## Architecture

There are two independent data flows that feed the system:

**Flow 1 — Real-world VD API (frontend display only)**
```
Taipei City VD API → tools/fetch_vd_data.py
                   → TrafficVision Design System/data/trafficData/*.json
                   → serve_api.py /api/traffic endpoint
```

**Flow 2 — Full ML pipeline (prediction + signal optimization)**
```
tools/grabapi.py (VD API)
  → tools/convertToRou.py + duarouter  (route generation)
  → VehicleData.py (SUMO batch simulation, 16 workers)
  → data/simulation_data/*.csv
  → train_model.py  (offline: GRU model training → gru_traffic_model.pth)
  → tools/runtime_pipeline.py:
      Step 1: tools/main.py (grab + route gen)
      Step 2: SUMO simulation → CSV
      Step 3: tools/predict_main.py → prediction CSV + signal override XML
      Step 4: tools/generate_edge_heatmap.py → edge_heatmap.json
      → data/runtime_data/<stem>/handoff/
```

**Frontend / API**
- `TrafficVision Design System/serve_api.py` — FastAPI server; reads from both data flows. Exposes REST endpoints and a WebSocket for live SUMO streaming.
- `docker-compose.yml` — runs `api` (FastAPI) + `ollama` (Gemma 4 LLM). Use this for production.

**LLM fine-tuning (offline, GPU required)**
- `tools/generate_finetune_dataset.py` → `data/finetune_dataset.jsonl`
- `tools/finetune_gemma.py` (Unsloth + TRL) → `tools/export_to_gguf.py`

## Key Files

| File | Role |
|------|------|
| `tools/runtime_pipeline.py` | Master orchestrator for one-shot or scheduled runs |
| `tools/predict_main.py` | `run_full_pipeline()` — prediction → signal optimization → handoff |
| `tools/traffic_light_optimizer.py` | Core signal-timing strategy; evaluates multiple strategies via SUMO |
| `tools/traffic_optimizer_signal.py` | Builds phase-override XML (`add.xml`) written to SUMO |
| `tools/traffic_optimizer_io.py` | SUMO I/O helpers (config creation, simulation runner, stats parsing) |
| `tools/predict_to_csv.py` | Loads `gru_traffic_model.pth`, produces prediction CSV |
| `tools/export_handoff.py` | Copies all outputs into `handoff/` with a manifest CSV |
| `train_model.py` | GRU model training (offline, run once after collecting simulation data) |
| `VehicleData.py` | Batch SUMO simulation with 16 parallel processes; requires `SUMO_HOME` |
| `TrafficVision Design System/serve_api.py` | FastAPI app; also embeds a TraCI WebSocket streaming endpoint |

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt          # API + pipeline runtime
pip install -r requirements-training.txt # Fine-tuning only (GPU required)
```

### Run the API server (development)
```bash
python "TrafficVision Design System/serve_api.py"
# Listening on http://localhost:8000
```

### Run the runtime pipeline
```bash
# Single run (grab data → simulate → predict → signal plan)
python tools/runtime_pipeline.py --once

# Continuous scheduler (default 300 s interval)
python tools/runtime_pipeline.py --interval 300

# Custom model path
python tools/runtime_pipeline.py --once --model-path path/to/model.pth
```

### Train the GRU model
```bash
python train_model.py
# Reads from data/simulation_data/  →  writes gru_traffic_model.pth
```

### Evaluate the model
```bash
python test_model.py
# Reads from data/simulation_data_check/  →  prints MAE report
```

### Run SUMO batch simulation (generates training data)
```bash
# Requires SUMO_HOME environment variable
set SUMO_HOME=C:\path\to\sumo     # Windows
python VehicleData.py
# Reads data/VehicleData/*.rou.xml → writes data/simulation_data/*.csv
```

### Docker (production)
```bash
docker-compose up          # starts ollama + api
docker-compose up --build  # rebuild after Dockerfile changes
```

### Generate LLM fine-tuning dataset
```bash
python tools/generate_finetune_dataset.py   # → data/finetune_dataset.jsonl
python tools/finetune_gemma.py              # requires GPU + requirements-training.txt
python tools/export_to_gguf.py             # exports to GGUF for Ollama
```

## Data Directories

| Path | Contents |
|------|----------|
| `data/VehicleData/` | SUMO route files (input to `VehicleData.py`) |
| `data/simulation_data/` | Per-run CSVs from batch simulation (training data) |
| `data/runtime_data/<stem>/handoff/` | Live pipeline outputs consumed by the API |
| `TrafficVision Design System/data/trafficData/` | Real-world VD JSON snapshots |

## Environment Requirements

- **SUMO_HOME** must be set to run `VehicleData.py` or any code that imports `traci`/`sumolib`.
- The API container does **not** need SUMO — it only reads files produced by the pipeline.
- Ollama service must be running at `http://localhost:11434` (or `$OLLAMA_BASE_URL`) for LLM features.

## Model Details

`gru_traffic_model.pth` is a PyTorch checkpoint containing `model_state_dict`, `edge_ids`, `scaler` (Log1p), and a `config` dict. The `GRUSequence` and `Log1pScaler` class definitions must be re-declared when loading the checkpoint outside `train_model.py` — see `test_model.py:register_legacy_checkpoint_classes()`.

- Input: 15 timesteps × (num_edges + 2 time features), 20 s per step
- Output: 15-step sequence (5 min horizon) per edge
- Loss: weighted MSE that up-weights high-traffic edges (>10 vehicles)

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **System-of-the-control-traffic-light** (6058 symbols, 8127 relationships, 145 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/System-of-the-control-traffic-light/context` | Codebase overview, check index freshness |
| `gitnexus://repo/System-of-the-control-traffic-light/clusters` | All functional areas |
| `gitnexus://repo/System-of-the-control-traffic-light/processes` | All execution flows |
| `gitnexus://repo/System-of-the-control-traffic-light/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

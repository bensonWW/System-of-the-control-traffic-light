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

## Data Sources Cheat Sheet — what each dashboard number actually means

The dashboard's three map modes (原始 / 預測 / 優化) and the cards above them
draw from **different physical quantities and different update cadences**.
Mixing them up causes long debugging sessions — fields with the same label
(`平均車速`, `總車流量`) can legitimately differ by 5-10× because the units
differ. Read this before adding new dashboard logic.

### Update cadence

| Source | Update mechanism | Typical age |
|--------|------------------|-------------|
| VD JSON (`trafficData/*.json`) | `_periodic_traffic_refresh` 自動 5-min | < 5 min |
| `*_predict.csv` (GRU output) | `runtime_pipeline.py` 跑完才更新 | depends on scheduler |
| `edge_heatmap*.json` | runtime_pipeline Step 4 一起寫 | same as predict |
| Pipeline scheduler | `python tools/runtime_pipeline.py` (no `--once`) | 預設 5-min interval |

If `*_predict.csv` is hours old, the scheduler isn't running. Start with
`python tools/runtime_pipeline.py --interval 300` or check `_metrics.jsonl`.
`/api/health` exposes all of the above ages.

### Numeric semantics (the part everyone gets wrong)

| Field | VD (原始) | GRU predict (預測) | SUMO best (優化) |
|-------|-----------|--------------------|---------------------|
| `vol` 物理意義 | 5-min **通過量** (pass-through flow) | Σ vehicle_count over 15 × 20-sec **snapshots** (occupancy sum) | **= 預測 vol** (vehicles preserved through optimization) |
| `vol` typical magnitude | 1,000-2,000 | 200-300 | same as 預測 |
| Conversion factor | — | ÷ ~6 by Little's Law (dwell_time/window) | — |
| `spd` source | VD `AvgSpd` | SUMO `no_control` baseline sim | SUMO `best-strategy` sim |
| `spd = 0` 意義 | 真實偵測到 0 km/h (極塞) | "SUMO 沒派車到這邊 → 無測量" (now returned as `null`) | same fallback as 預測 |
| `occ` source | VD `AvgOcc` (%) | SUMO baseline occupancy | SUMO best-strategy occupancy |
| MOE | VD 直接給 | 從 spd 推算 (`_moe_from_spd`) | same |

**Why 預測 ≠ 優化 only in spd/occ, not in vol**: optimization redistributes
signal timing, it doesn't make vehicles appear or disappear. `opt.vol` is
explicitly set to `pred.vol` in `serve_api.py:get_road_forecast` /
`get_edge_forecast`. The optimization effect shows in spd/occ only.

**Why 預測 and 優化 sometimes show identical values across spd/occ too**:
when `traffic_light_optimizer` picks `no_control` as the winning strategy
(`composite_score = 1.0`, no strategy beat baseline), the "best strategy"
edgedata == baseline edgedata, so they are literally the same file.
This is correct behavior — the dashboard banner shows the strategy choice.

**Dashboard's `平均車速` filter**: only roads with `spd > 0` are counted
(commit `d95dd1be5`). Excludes ~19 dead-end edges where SUMO sim had no
vehicles. Without this filter, pred-mode 平均車速 was diluted from ~41 to
~32 km/h by those zero entries.

**Dashboard's `總車流量`**: sum of table rows' `vol`. Because the unit
families differ (VD = pass-through; GRU/SUMO = snapshot sum), comparing
these totals across modes is misleading — 1,500 vs 250 is the same traffic
load measured two different ways, not a 6× drop.

### Tunable constants (env vars)

Added in this stabilization round; defaults preserve original behavior.

| Env var | Default | Effect | Where |
|---------|---------|--------|-------|
| `TRAFFICVISION_VD_WORKERS` | `min(16, cpu_count())` | SUMO batch sim worker pool | `VehicleData.py` |
| `TRAFFICVISION_STRATEGY_TIMEOUT` | `300` | Per-strategy SUMO eval budget (s) | `tools/traffic_light_optimizer.py` |
| `TRAFFICVISION_MIN_YELLOW` | `3.0` | Min yellow phase duration (SUMO physics floor) | `tools/traffic_optimizer_signal.py` |
| `TRAFFICVISION_MIN_ALL_RED` | `1.0` | Min all-red clearance | same |
| `TRAFFICVISION_MIN_GREEN` | `5.0` | Min green phase duration | same |
| `TRAFFICVISION_VOL_DECAY` | `0.75` | OD route flow decay per intermediate edge | `tools/fixRoadData.py` |
| `TRAFFICVISION_BACKFILL_ALPHA` | `0.35` | Recursive backfill weight for missing edge flow | same |
| `TRAFFICVISION_TRIP_MAX_EDGES` | `10` | DFS max edges per synthesized trip | same |
| `TRAFFICVISION_WS_TOKEN` | (unset) | If set, `/ws/simulation` requires bearer token | `serve_api.py` |
| `FT_MAX_TRAFFIC_FILES` | `200` | Fine-tune dataset: max VD JSON sampled | `tools/generate_finetune_dataset.py` |
| `FT_MAX_HANDOFF_DIRS` | `30` | Fine-tune dataset: max handoff dirs sampled | same |

### Diagnostic endpoints

- `GET /api/status` — quick: latest VD + handoff file paths
- `GET /api/health` — deep: VD/predict/scheduler/Ollama ages + `status: ok|degraded|critical`
- `data/runtime_data/_metrics.jsonl` — one JSON line per `run_once()`, append-only history

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **System-of-the-control-traffic-light** (7037 symbols, 9623 relationships, 185 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
| Work in the Unsloth_compiled_cache area (382 symbols) | `.claude/skills/generated/unsloth-compiled-cache/SKILL.md` |
| Work in the Tools area (119 symbols) | `.claude/skills/generated/tools/SKILL.md` |
| Work in the TrafficVision Design System area (43 symbols) | `.claude/skills/generated/trafficvision-design-system/SKILL.md` |
| Work in the Cluster_48 area (13 symbols) | `.claude/skills/generated/cluster-48/SKILL.md` |
| Work in the Cluster_11 area (7 symbols) | `.claude/skills/generated/cluster-11/SKILL.md` |
| Work in the Cluster_9 area (6 symbols) | `.claude/skills/generated/cluster-9/SKILL.md` |
| Work in the Cluster_12 area (4 symbols) | `.claude/skills/generated/cluster-12/SKILL.md` |

<!-- gitnexus:end -->

# Traffic Data Integration for SUMO

This project integrates real-world traffic signal timing data (from Taipei City Open Data) into a SUMO traffic simulation network (`ntut-the way.net.xml`).

## Project Overview

The pipeline consists of three main steps:
1.  **Data Integration**: Merges raw JSON and CSV data into a clean, unified format.
2.  **Spatial Matching**: Maps SUMO simulation nodes to real-world intersections based on GPS coordinates.
3.  **Simulation Configuration**: Generates SUMO traffic light logic (`traffic_light.add.xml`) based on the timing plans.

## Files & Scripts

### Scripts
| File | Description |
| :--- | :--- |
| `integrate_traffic_data.py` | Parsers raw `timing_plan.json` (with repair logic for corruption), `timing_plan_table.json`, and CSV location data. Exports `integrated_timing_plan.json`. |
| `compare_traffic_data.py` | Compares SUMO Network coordinates with the integrated JSON data. Generates the mapping file `sumo_json_mapping.csv`. |
| `apply_sumo_timings.py` | Reads the mapping and timing data to generate `traffic_light.add.xml`. Handles complex "Joined" traffic lights and heuristic phase mapping. |

### Data Files
| File | Description |
| :--- | :--- |
| `integrated_timing_plan.json` | The cleaned, merged traffic data source. |
| `sumo_json_mapping.csv` | The generated mapping table between SUMO Junction IDs and Real-World ICIDs. |
| `traffic_light.add.xml` | The final SUMO additional file containing the `<tlLogic>` definitions. |

## Usage Guide

### 1. Run Data Integration
If you need to regenerate the source data:
```bash
python integrate_traffic_data.py
```

### 2. Run Spatial Matching
To verify the coordinate matching between SUMO and Real World:
```bash
python compare_traffic_data.py
```
*   This will update `sumo_json_mapping.csv`.

### 3. Generate SUMO Configuration
To create the traffic light logic file:
```bash
python apply_sumo_timings.py
```
*   Output: `traffic_light.add.xml`

### 4. Load in SUMO
Edit your `.sumocfg` file to include the generated XML:

```xml
<configuration>
    <input>
        <net-file value="ntut-the way.net.xml"/>
        <additional-files value="traffic_light.add.xml"/>
    </input>
</configuration>
```

## Notes
-   **Coverage**: The system currently matches **30** real-world intersections.
-   **Clusters**: Due to SUMO's "Joined TLS" feature, these 30 intersections correspond to **15** traffic light controllers in the simulation.

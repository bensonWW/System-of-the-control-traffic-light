# Traffic Control System Simulation

This project simulates traffic flow in the NTUT area of Taipei using real-time data from the Taipei City Open Data API. It combines real-world traffic volume with **SUMO (Simulation of Urban MObility)** to create a dynamic traffic simulation.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.x** installed.
- **SUMO** installed and added to your system `PATH`. [Install SUMO](https://sumo.dlr.de/docs/Downloads.php)

### 2. Installation
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Running the Simulation

#### **Step 1: Collect & Process Traffic Data**
Run the main data collection script to fetch real-time data, process it, and generate SUMO route files.
```bash
python scripts/collect_traffic_data.py
```
- **Raw Data Output**: `data/trafficData/` (JSON format)
- **SUMO Route Output**: `data/DDATA/` (SUMO .rou.xml format)

#### **Step 2: Start SUMO Simulation**
Open the main simulation configuration in SUMO-GUI.
```bash
sumo-gui -c data/ntut_config.sumocfg
```

---

## 📂 Project Structure & Tools

### Core Scripts

#### `scripts/collect_traffic_data.py`
The main entry point for data collection. It orchestrates the entire pipeline:
1.  Downloads real-time traffic data (XML) from the Open Data API.
2.  Saves raw data to `data/trafficData/` (e.g., `traffic_data_YYYYMMDD_HHMMSS.json`).
3.  Calls various tools (see below) to filter, process, and generate SUMO route files.
4.  Saves the final route file to `data/DDATA/` (e.g., `traffic_data_YYYYMMDD_HHMMSS.rou.xml`).

### Tools (`/tools`)

| Tool File | Description |
| :--- | :--- |
| **`grabapi.py`** | **Data Fetcher**: Downloads the `GetVD.xml.gz` file from the Taipei City Open Data API, decompresses it, and parses the XML to extract traffic volume and speed data. |
| **`selectRoad.py`** | **Data Filter**: Filters the raw traffic data based on the map's boundary. It uses `searchnetdata.py` to get the map's coordinate range and retains only the roads that fall within this range (using XY coordinates). |
| **`searchnetdata.py`** | **Network Query**: Parses the SUMO network file (`.net.xml`) to determine the map boundary (lat/lon) and provides functions to search for edge IDs based on coordinates. |
| **`convertToRou.py`** | **Route Generator**: Converts the processed traffic data into SUMO route files (`.rou.xml`). It uses SUMO's `duarouter` to calculate valid routes on the network based on the traffic volume. |
| **`fixRoadData.py`** | **Data Imputation**: Handles missing or incomplete traffic data by imputing values based on historical or neighboring road data to ensure the simulation runs smoothly. |
| **`apply_sumo_timings.py`**| **Traffic Light Control**: Generates the traffic light timing configuration file (`traffic_light.add.xml`) based on the defined timing plans. |
| **`connections_out.py`** | **Network Analysis**: Extracts and analyzes connection data from the network file to understand lane-to-lane connectivity. |

### Data Directory (`/data`)

- **`ntut_config.sumocfg`**: Main SUMO configuration file.
- **`ntut_network_split.net.xml`**: The primary road network file used for simulation.
- **`trafficData/`**: Stores raw traffic data downloaded from the API (JSON).
- **`DDATA/`**: Stores the processed SUMO route files (`.rou.xml`).
- **`source/`**: Contains original source data (e.g., raw OSM maps).
- **`legacy/`**: Backup of older network files.

---

## 🛠️ Configuration

- **Map Boundary**: Defined dynamically in `ntut-the way.net.xml` (or `ntut_network_split.net.xml`) and read by `searchnetdata.py`.
- **API URL**: Configured in `scripts/collect_traffic_data.py`.
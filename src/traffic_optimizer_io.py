import os
import sys
import urllib.parse
import xml.etree.ElementTree as ET
import glob

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  # type: ignore
import sumolib  # type: ignore

PATH_TAGS = {
    "net-file",
    "route-files",
    "additional-files",
    "tripinfo-output",
    "statistic-output",
    "gui-settings-file",
}


def _resolve_path(base_cfg, value):
    if not value:
        return value
    value = urllib.parse.unquote(value.strip())
    if not value or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_cfg)), value))


def _resolve_path_list(base_cfg, value):
    if not value:
        return []
    return [_resolve_path(base_cfg, item) for item in value.split(",") if item.strip()]


def _normalize_config_paths(base_cfg, root):
    for node in root.iter():
        value = node.get("value")
        if value is None or node.tag not in PATH_TAGS:
            continue
        if node.tag in {"route-files", "additional-files"}:
            node.set("value", ",".join(_resolve_path_list(base_cfg, value)))
        else:
            node.set("value", _resolve_path(base_cfg, value))


def _apply_output_overrides(root, output_overrides=None):
    if not output_overrides:
        return

    output_node = root.find("output")
    if output_node is None:
        output_node = ET.SubElement(root, "output")

    for tag_name, file_path in output_overrides.items():
        node = output_node.find(tag_name)
        if node is None:
            node = ET.SubElement(output_node, tag_name)
        # Path-like output tags must be absolute, but scalar tags (e.g. output-prefix)
        # must keep raw values; otherwise SUMO builds invalid output paths.
        if tag_name in PATH_TAGS:
            node.set("value", os.path.abspath(file_path))
        else:
            node.set("value", str(file_path))


def create_temp_sumo_cfg(route_file, base_cfg, temp_cfg_path, additional_files=None, output_overrides=None, exclude_additional_basenames=None):
    try:
        tree = ET.parse(base_cfg)
        root = tree.getroot()
        _normalize_config_paths(base_cfg, root)
        _apply_output_overrides(root, output_overrides)

        input_node = root.find("input")
        route_node = input_node.find("route-files") if input_node is not None else None
        if route_node is None:
            raise AttributeError("Missing route-files node in sumocfg.")

        route_node.set("value", os.path.abspath(route_file))

        if additional_files is not None:
            add_node = input_node.find("additional-files")
            existing = _resolve_path_list(base_cfg, add_node.get("value", "")) if add_node is not None else []
            if exclude_additional_basenames:
                drop = {name.lower() for name in exclude_additional_basenames}
                existing = [p for p in existing if os.path.basename(p).lower() not in drop]
            merged = existing + [os.path.abspath(p) for p in additional_files if p]
            if add_node is None:
                add_node = ET.SubElement(input_node, "additional-files")
            add_node.set("value", ",".join(merged))

        tree.write(temp_cfg_path)
        return True
    except AttributeError:
        print("Error: Invalid sumocfg structure.")
        return False


def filter_route_file(input_rou, output_rou, valid_edges):
    try:
        tree = ET.parse(input_rou)
        root = tree.getroot()
        valid_vehicles = []

        for vehicle in root.findall("vehicle"):
            route = vehicle.find("route")
            if route is not None:
                edges = route.get("edges", "").split()
                if edges and all(edge in valid_edges for edge in edges):
                    valid_vehicles.append(vehicle)
                continue

            route_distribution = vehicle.find("routeDistribution")
            if route_distribution is None:
                continue

            routes = route_distribution.findall("route")
            if routes and all(
                (route.get("edges", "").split() and all(edge in valid_edges for edge in route.get("edges", "").split()))
                for route in routes
            ):
                valid_vehicles.append(vehicle)

        kept_children = [child for child in root if child.tag != "vehicle" or child in valid_vehicles]
        root.clear()
        root.tag = "routes"
        for child in kept_children:
            root.append(child)

        tree.write(output_rou)
        return len(valid_vehicles)
    except Exception as e:
        print(f"Filter error {input_rou}: {e}")
        return 0


def summarize_stats_xml(stats_xml_path):
    if not os.path.exists(stats_xml_path):
        base_dir = os.path.dirname(stats_xml_path)
        base_name = os.path.basename(stats_xml_path)
        candidates = sorted(glob.glob(os.path.join(base_dir, f"*{base_name}")))
        if candidates:
            stats_xml_path = candidates[-1]
        else:
            return {}

    root = ET.parse(stats_xml_path).getroot()
    performance = root.find("performance")
    teleports = root.find("teleports")
    trip_stats = root.find("vehicleTripStatistics")
    vehicles = root.find("vehicles")

    return {
        "simulation_end_time": float(performance.get("end", 0.0)) if performance is not None else 0.0,
        "vehicle_count": int(
            trip_stats.get("count", vehicles.get("inserted", 0))
            if trip_stats is not None and vehicles is not None
            else trip_stats.get("count", 0)
            if trip_stats is not None
            else vehicles.get("inserted", 0)
            if vehicles is not None
            else 0
        ),
        "avg_duration": float(trip_stats.get("duration", 0.0)) if trip_stats is not None else 0.0,
        "avg_waiting_time": float(trip_stats.get("waitingTime", 0.0)) if trip_stats is not None else 0.0,
        "avg_time_loss": float(trip_stats.get("timeLoss", 0.0)) if trip_stats is not None else 0.0,
        "avg_depart_delay": float(trip_stats.get("departDelay", 0.0)) if trip_stats is not None else 0.0,
        "teleports_total": int(teleports.get("total", 0)) if teleports is not None else 0,
        "teleports_wrong_lane": int(teleports.get("wrongLane", 0)) if teleports is not None else 0,
    }


def run_sumo_simulation_with_end_time(config_file, output_csv=None, override_program_map=None):
    cmd = [
        sumolib.checkBinary("sumo"),
        "-c", config_file,
        "--start",
        "--quit-on-end",
        "--no-warnings",
    ]

    # Explicit free port + numRetries: 5 strategy workers + the orchestrator's
    # current-demand sim can collide on traci's default port picker. Asking the
    # OS for a free port + retrying narrows the race window.
    port = sumolib.miscutils.getFreeSocketPort()
    label = f"opt_{os.getpid()}_{port}"
    try:
        traci.start(cmd, port=port, label=label, numRetries=10)
        traci.switch(label)
        applied_schedule_times = set()
        data_buffer = []
        simulation_end_time = 0.0

        if override_program_map:
            for schedule_time in sorted(override_program_map):
                if schedule_time <= 0:
                    for tl_id, program_id in override_program_map[schedule_time].items():
                        try:
                            traci.trafficlight.setProgram(tl_id, program_id)
                        except Exception:
                            pass
                    applied_schedule_times.add(schedule_time)

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            simulation_end_time = current_time

            if override_program_map:
                current_key = int(round(current_time))
                if current_key in override_program_map and current_key not in applied_schedule_times:
                    for tl_id, program_id in override_program_map[current_key].items():
                        try:
                            traci.trafficlight.setProgram(tl_id, program_id)
                        except Exception:
                            pass
                    applied_schedule_times.add(current_key)

            if output_csv and current_time % 20 == 0:
                for edge in traci.edge.getIDList():
                    count = traci.edge.getLastStepVehicleNumber(edge)
                    if count > 0:
                        # 同步採集平均速度供 v2 pair model (count + speed) 使用。
                        # 照 VehicleData.py L117-125 的做法: 只在 count > 0 時取,
                        # 避免 TraCI 對空 edge 回特殊值 (-1001 等)。單位 m/s → km/h。
                        mean_speed_ms = traci.edge.getLastStepMeanSpeed(edge)
                        avg_speed_kmh = mean_speed_ms * 3.6
                        data_buffer.append((current_time, edge, count, avg_speed_kmh))

        if output_csv:
            actual_end_time = data_buffer[-1][0] if data_buffer else simulation_end_time
            # Adaptive trim: pair-based model needs >= 15 unique time bins to do
            # inference. Default policy is leading 60s warmup + trailing 100s
            # (covers ~80% of normal cases where end_time >= 460s). For low-traffic
            # nights the sim ends fast (e.g. end_time ~ 260s), and a fixed 100s
            # tail trim leaves only 8 bins — predict_to_csv then raises.
            # Strategy: keep the 60s warmup as-is (vehicles entering = unreliable
            # data), but shrink the tail to whatever's necessary to retain 15
            # bins. If the sim is so short that even no-trim can't give 15 bins,
            # we still emit what we have and let the upstream caller decide
            # (predict will raise; runtime_pipeline catches it).
            LEADING_WARMUP = 60
            REQUIRED_BINS = 15
            BIN_SEC = 20  # SUMO data emit interval
            DEFAULT_TAIL_TRIM = 100
            needed_window = REQUIRED_BINS * BIN_SEC  # 300s
            available_after_warmup = max(0.0, actual_end_time - LEADING_WARMUP)
            if available_after_warmup >= needed_window + DEFAULT_TAIL_TRIM:
                tail_trim = DEFAULT_TAIL_TRIM
            else:
                # Shrink tail trim to whatever still keeps >= needed_window
                tail_trim = max(0.0, available_after_warmup - needed_window)
            cutoff_upper = actual_end_time - tail_trim
            # 4 欄 CSV (新增 avg_speed_kmh 給 v2 pair model)。下游 predict_to_csv 用具名
            # 欄位讀,多 1 欄相容。VehicleData.py 已是同格式。
            with open(output_csv, "w", encoding="utf-8") as f:
                f.write("time,edge_id,vehicle_count,avg_speed_kmh\n")
                f.writelines(
                    f"{current_time},{edge},{count},{speed:.2f}\n"
                    for current_time, edge, count, speed in data_buffer
                    if LEADING_WARMUP <= current_time <= cutoff_upper
                )

        return True, simulation_end_time
    except Exception:
        return False, 0.0
    finally:
        try:
            traci.close()
        except Exception:
            pass
        sys.stdout.flush()


def get_valid_edge_ids(net_file):
    return {
        edge.get("id")
        for edge in ET.parse(net_file).getroot().findall("edge")
        if edge.get("id") and not edge.get("id").startswith(":")
    }


def get_source_stem_from_prediction_csv(prediction_csv):
    stem = os.path.splitext(os.path.basename(prediction_csv))[0]
    return stem[:-8] if stem.endswith("_predict") else stem


def find_route_xml_for_prediction(prediction_csv, temp_route_dir):
    route_xml = os.path.join(temp_route_dir, f"{get_source_stem_from_prediction_csv(prediction_csv)}.rou.xml")
    if not os.path.exists(route_xml):
        raise FileNotFoundError(f"找不到對應的 route XML: {route_xml}")
    return route_xml

import os
import sys
import shutil
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  # type: ignore
import sumolib  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
BASE_SUMOCFG = os.path.join(ROOT_DIR, "data", "ntut-the way.sumocfg")
NET_FILE = os.path.join(ROOT_DIR, "data", "ntut-the way.net.xml")
TEMP_ROUTE_DIR = os.path.join(ROOT_DIR, "data", "temp")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT_DIR, "data", "prediction_runs")

DEFAULT_TOP_N_TLS = 6
DEFAULT_TOP_EDGE_COUNT = 20
TOP_PHASES_TO_ADJUST = 2
PRIMARY_PHASE_EXTRA_SECONDS = 10
SECONDARY_PHASE_EXTRA_SECONDS = 6
MIN_EDGE_SCORE = 0.01

FROM_EDGE_WEIGHT = 1.0
TO_EDGE_WEIGHT = 0.8
EDGE_SCORE_EXPONENT = 1.35

DEFAULT_STRATEGY = {
    "name": "default",
    "top_n_tls": DEFAULT_TOP_N_TLS,
    "top_phases_to_adjust": TOP_PHASES_TO_ADJUST,
    "primary_phase_extra": PRIMARY_PHASE_EXTRA_SECONDS,
    "secondary_phase_extra": SECONDARY_PHASE_EXTRA_SECONDS,
}

UNSAFE_TLS_IDS = {
    "joinedS_3086736519_3086736520_631668954_631668971_#3more",
    "GS_cluster_9383263990_9383263991_9383263992_9383263993",
    "cluster_2038168688_655375573",
    "cluster_2528018119_655375236",
    "cluster_3431431575_4777363357_655375231_6982059025",
    "cluster_623980090_623980094",
}

STATE_MAP = {
    "G": "綠燈",
    "g": "綠燈(次要)",
    "y": "黃燈",
    "Y": "黃燈(優先)",
    "r": "紅燈",
    "R": "紅燈(優先)",
    "-": "無控制",
    "s": "停止",
    "u": "未知",
}

PATH_TAGS = {
    "net-file",
    "route-files",
    "additional-files",
    "tripinfo-output",
    "statistic-output",
    "gui-settings-file",
}

METRICS = [
    "simulation_end_time",
    "vehicle_count",
    "avg_duration",
    "avg_waiting_time",
    "avg_time_loss",
    "avg_depart_delay",
    "teleports_total",
    "teleports_wrong_lane",
]


def resolve_strategy(strategy=None):
    return {**DEFAULT_STRATEGY, **(strategy or {})}


def decode_traffic_light_state(state):
    return " | ".join(
        f"信號{i + 1}:{STATE_MAP.get(ch, f'未知({ch})')}"
        for i, ch in enumerate(state)
    )


def _format_num(value):
    if isinstance(value, int):
        return str(value)
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


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
        node.set("value", os.path.abspath(file_path))


def extract_traffic_lights(_, tree):
    traffic_lights = []
    for tl_logic in tree.getroot().findall("tlLogic"):
        phases = []
        for phase in tl_logic.findall("phase"):
            duration = float(phase.get("duration", "0"))
            state = phase.get("state", "N/A")
            phases.append({
                "duration": duration,
                "state": state,
                "minDur": float(phase.get("minDur", duration)),
                "maxDur": float(phase.get("maxDur", duration)),
                "hasMinDur": phase.get("minDur") is not None,
                "hasMaxDur": phase.get("maxDur") is not None,
                "description": decode_traffic_light_state(state),
            })

        traffic_lights.append({
            "id": tl_logic.get("id", "N/A"),
            "type": tl_logic.get("type", "N/A"),
            "programID": tl_logic.get("programID", "0"),
            "offset": float(tl_logic.get("offset", "0")),
            "phases": phases,
            "total_cycle_time": sum(p["duration"] for p in phases),
            "phase_count": len(phases),
        })
    return traffic_lights


def extract_edge_names(tree):
    return {
        edge.get("id", "N/A"): edge.get("name", "")
        for edge in tree.getroot().findall("edge")
    }


def extract_connections(tree, exclude_internal=False):
    edge_names = extract_edge_names(tree)
    connections = []

    for conn in tree.getroot().findall("connection"):
        from_edge = conn.get("from", "N/A")
        to_edge = conn.get("to", "N/A")

        if exclude_internal and from_edge.startswith(":"):
            continue

        from_name = edge_names.get(from_edge, "")
        to_name = edge_names.get(to_edge, "")

        connections.append({
            "from_edge": f"'{from_edge}" if from_edge.startswith("-") else from_edge,
            "to_edge": f"'{to_edge}" if to_edge.startswith("-") else to_edge,
            "from_edge_name": from_name,
            "to_edge_name": to_name,
            "from_lane": conn.get("fromLane", "N/A"),
            "to_lane": conn.get("toLane", "N/A"),
            "via": conn.get("via", "N/A"),
            "direction": conn.get("dir", "N/A"),
            "state": conn.get("state", "N/A"),
            "type": "交叉路口內部" if from_edge.startswith(":") else "道路連接",
            "description": (
                f"從 {from_name or from_edge} 路的第 {conn.get('fromLane', 'N/A')} 車道"
                f"連接到 {to_name or to_edge} 路的第 {conn.get('toLane', 'N/A')} 車道"
            ),
            "controlled_by_tl": False,
            "tl_id": None,
            "tl_link_index": None,
        })

    return connections


def link_connections_to_traffic_lights(connections, _, tree):
    conn_map = {
        (
            c["from_edge"].strip("'"),
            c["to_edge"].strip("'"),
            c["from_lane"],
            c["to_lane"],
        ): c
        for c in connections
    }

    for conn in tree.getroot().findall("connection"):
        tl_id = conn.get("tl")
        link_index = conn.get("linkIndex")
        if not tl_id or link_index is None:
            continue

        key = (
            conn.get("from", "N/A"),
            conn.get("to", "N/A"),
            conn.get("fromLane", "N/A"),
            conn.get("toLane", "N/A"),
        )
        if key in conn_map:
            conn_map[key]["controlled_by_tl"] = True
            conn_map[key]["tl_id"] = tl_id
            conn_map[key]["tl_link_index"] = int(link_index)

    return connections


def create_temp_sumo_cfg(route_file, base_cfg, temp_cfg_path, additional_files=None, output_overrides=None):
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


def get_phase_duration_bounds(phase):
    base_duration = float(phase["duration"])
    min_duration = float(phase.get("minDur", base_duration))
    max_duration = float(phase.get("maxDur", base_duration))

    if not phase.get("hasMinDur"):
        min_duration = min(min_duration, max(3.0, base_duration * 0.4))
    if not phase.get("hasMaxDur"):
        max_duration = max(max_duration, base_duration + 20.0, base_duration * 3.0)

    return min_duration, max(min_duration, max_duration)


def clamp_phase_duration(phase, new_duration):
    min_duration, max_duration = get_phase_duration_bounds(phase)
    return max(min_duration, min(max_duration, float(new_duration)))


def summarize_stats_xml(stats_xml_path):
    if not os.path.exists(stats_xml_path):
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

    try:
        traci.start(cmd)
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
                        data_buffer.append((current_time, edge, count))

        if output_csv:
            actual_end_time = data_buffer[-1][0] if data_buffer else simulation_end_time
            with open(output_csv, "w", encoding="utf-8") as f:
                f.write("time,edge_id,vehicle_count\n")
                f.writelines(
                    f"{current_time},{edge},{count}\n"
                    for current_time, edge, count in data_buffer
                    if 60 <= current_time <= (actual_end_time - 100)
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


def get_valid_edge_ids(net_file=NET_FILE):
    return {
        edge.get("id")
        for edge in ET.parse(net_file).getroot().findall("edge")
        if edge.get("id") and not edge.get("id").startswith(":")
    }


def get_source_stem_from_prediction_csv(prediction_csv):
    stem = os.path.splitext(os.path.basename(prediction_csv))[0]
    return stem[:-8] if stem.endswith("_predict") else stem


def find_route_xml_for_prediction(prediction_csv):
    route_xml = os.path.join(TEMP_ROUTE_DIR, f"{get_source_stem_from_prediction_csv(prediction_csv)}.rou.xml")
    if not os.path.exists(route_xml):
        raise FileNotFoundError(f"找不到對應的 route XML: {route_xml}")
    return route_xml


def normalize_predicted_edge_scores(edge_scores, top_edge_count=DEFAULT_TOP_EDGE_COUNT):
    filtered = [
        (str(edge_id).strip("'"), float(score))
        for edge_id, score in edge_scores.items()
        if not str(edge_id).strip("'").startswith(":") and float(score) >= MIN_EDGE_SCORE
    ]
    if not filtered:
        return {}

    filtered.sort(key=lambda item: item[1], reverse=True)
    selected = filtered[:top_edge_count]
    max_score = selected[0][1]
    if max_score <= 0:
        return {}

    selected_count = len(selected)
    normalized = {}

    for rank, (edge_id, score) in enumerate(selected):
        relative = score / max_score
        smoothed = ((relative ** 0.75) * 0.7 + relative * 0.3)
        capped = min(smoothed, 0.85 if rank == 0 else 0.80)
        rank_boost = 1.0 + max(0.0, (selected_count - rank - 1) * 0.08)
        normalized[edge_id] = capped * rank_boost

    return normalized


def build_tls_scores_from_predictions(connections, edge_scores, strategy=None):
    tls_scores = defaultdict(float)
    tls_links = defaultdict(set)
    link_scores = defaultdict(float)

    top_edge_count = DEFAULT_TOP_EDGE_COUNT
    if strategy is not None:
        top_edge_count = int(strategy.get("top_edge_count_override", DEFAULT_TOP_EDGE_COUNT))

    normalized_edge_scores = normalize_predicted_edge_scores(
        edge_scores,
        top_edge_count=top_edge_count,
    )

    if not normalized_edge_scores:
        return tls_scores, tls_links, link_scores

    for conn in connections:
        if not conn.get("controlled_by_tl") or conn["tl_id"] in UNSAFE_TLS_IDS:
            continue

        from_edge = conn["from_edge"].strip("'")
        to_edge = conn["to_edge"].strip("'")
        score = (
            normalized_edge_scores.get(from_edge, 0.0) * FROM_EDGE_WEIGHT
            - normalized_edge_scores.get(to_edge, 0.0) * TO_EDGE_WEIGHT
        )

        if score <= 0:
            continue

        tl_id = conn["tl_id"]
        link_index = int(conn["tl_link_index"])
        tls_scores[tl_id] += score
        tls_links[tl_id].add(link_index)
        link_scores[(tl_id, link_index)] += score

    return tls_scores, tls_links, link_scores


def _phase_green_links(phases, required_links):
    result = []
    for phase in phases:
        state = phase["state"]
        if "y" in state or "Y" in state:
            result.append([])
            continue
        result.append([idx for idx in required_links if idx < len(state) and state[idx] in {"G", "g"}])
    return result


def build_signal_plan_from_edge_scores(traffic_lights, connections, edge_scores, top_n_tls=DEFAULT_TOP_N_TLS, strategy=None):
    strategy = resolve_strategy(strategy)
    traffic_light_map = {tl["id"]: tl for tl in traffic_lights}

    tls_scores, tls_links, link_scores = build_tls_scores_from_predictions(
        connections,
        edge_scores,
        strategy,
    )

    selected_tl_ids = [
        tl_id
        for tl_id, _ in sorted(
            tls_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:int(strategy.get("top_n_tls", top_n_tls))]
    ]

    signal_plan = {}
    selected_summary = []

    for tl_id in selected_tl_ids:
        tl = traffic_light_map.get(tl_id)
        required_links = tls_links.get(tl_id, set())
        if tl is None or not required_links:
            continue

        current_link_scores = {
            link_idx: score
            for (score_tl_id, link_idx), score in link_scores.items()
            if score_tl_id == tl_id
        }
        if sum(current_link_scores.values()) <= 0:
            continue

        phase_scores = []
        for phase_index, green_links in enumerate(_phase_green_links(tl["phases"], required_links)):
            phase_score = sum(current_link_scores.get(link_idx, 0.0) for link_idx in green_links)
            if phase_score > 0:
                phase_scores.append((phase_index, phase_score))

        if not phase_scores:
            continue

        phase_scores.sort(key=lambda item: item[1], reverse=True)
        updated_phases = []

        if strategy.get("is_proportional", False):
            total_phase_score = sum(score for _, score in phase_scores)
            phase_score_map = dict(phase_scores)
            proportional_pool = float(strategy.get("proportional_pool", 20.0))

            for phase_index, phase in enumerate(tl["phases"]):
                new_duration = float(phase["duration"])
                if (
                    total_phase_score > 0
                    and ("G" in phase["state"] or "g" in phase["state"])
                    and phase_score_map.get(phase_index, 0) > 0
                ):
                    new_duration += (phase_score_map[phase_index] / total_phase_score) * proportional_pool

                updated_phases.append({
                    "duration": clamp_phase_duration(phase, new_duration),
                    "state": phase["state"],
                    "minDur": phase.get("minDur"),
                    "maxDur": phase.get("maxDur"),
                    "hasMinDur": phase.get("hasMinDur"),
                    "hasMaxDur": phase.get("hasMaxDur"),
                })

        elif strategy.get("preserve_cycle", False) and len(phase_scores) >= 2:
            busiest_phase = phase_scores[0][0]
            least_busy_phase = phase_scores[-1][0]
            primary_extra = max(0.0, float(strategy.get("primary_phase_extra", PRIMARY_PHASE_EXTRA_SECONDS)))

            for phase_index, phase in enumerate(tl["phases"]):
                delta = (
                    primary_extra if phase_index == busiest_phase
                    else -primary_extra if phase_index == least_busy_phase
                    else 0.0
                )
                updated_phases.append({
                    "duration": clamp_phase_duration(phase, float(phase["duration"]) + delta),
                    "state": phase["state"],
                    "minDur": phase.get("minDur"),
                    "maxDur": phase.get("maxDur"),
                    "hasMinDur": phase.get("hasMinDur"),
                    "hasMaxDur": phase.get("hasMaxDur"),
                })

        else:
            top_count = max(1, int(strategy.get("top_phases_to_adjust", TOP_PHASES_TO_ADJUST)))
            selected_phase_indices = {phase_index for phase_index, _ in phase_scores[:top_count]}
            best_phase = phase_scores[0][0]
            primary_extra = max(0.0, float(strategy.get("primary_phase_extra", PRIMARY_PHASE_EXTRA_SECONDS)))
            secondary_extra = max(0.0, float(strategy.get("secondary_phase_extra", SECONDARY_PHASE_EXTRA_SECONDS)))

            for phase_index, phase in enumerate(tl["phases"]):
                if phase_index == best_phase and phase_index in selected_phase_indices:
                    delta = primary_extra
                elif phase_index in selected_phase_indices:
                    delta = secondary_extra
                else:
                    delta = 0.0

                updated_phases.append({
                    "duration": clamp_phase_duration(phase, float(phase["duration"]) + delta),
                    "state": phase["state"],
                    "minDur": phase.get("minDur"),
                    "maxDur": phase.get("maxDur"),
                    "hasMinDur": phase.get("hasMinDur"),
                    "hasMaxDur": phase.get("hasMaxDur"),
                })

        signal_plan[tl_id] = {
            "traffic_light": tl,
            "phases": updated_phases,
        }
        selected_summary.append((tl_id, tls_scores[tl_id]))

    return signal_plan, selected_summary


def get_signal_plan_from_prediction(prediction_df, top_n_tls, strategy):
    tree = ET.parse(NET_FILE)
    traffic_lights = extract_traffic_lights(NET_FILE, tree)
    connections = extract_connections(tree, exclude_internal=True)
    connections = link_connections_to_traffic_lights(connections, traffic_lights, tree)

    strategy = resolve_strategy(strategy)
    base_prediction_time = float(prediction_df["time"].min())
    update_interval = float(strategy.get("update_interval", 120.0))

    prediction_df = prediction_df.copy()
    prediction_df["time_bin"] = (
        ((prediction_df["time"] - base_prediction_time) // update_interval) * update_interval
        + base_prediction_time
    )

    time_signal_plans = []
    summary_rows = []

    for current_bin, time_df in prediction_df.groupby("time_bin", sort=True):
        apply_time = max(0.0, float(current_bin) - base_prediction_time)
        edge_scores = defaultdict(float)

        for row in time_df.itertuples(index=False):
            edge_scores[str(row.edge_id)] += float(row.vehicle_count)

        signal_plan, selected_summary = build_signal_plan_from_edge_scores(
            traffic_lights,
            connections,
            edge_scores,
            top_n_tls=top_n_tls,
            strategy=strategy,
        )

        if not signal_plan:
            continue

        time_signal_plans.append({
            "time": float(current_bin),
            "apply_time": apply_time,
            "signal_plan": signal_plan,
            "selected_summary": selected_summary,
        })

        summary_rows.extend({
            "time": float(current_bin),
            "apply_time": apply_time,
            "tl_id": tl_id,
            "score": float(score),
        } for tl_id, score in selected_summary)

    return time_signal_plans, pd.DataFrame(summary_rows)


def _make_override_program_id(base_program_id, tl_id, schedule_time):
    safe_tl_id = str(tl_id).replace("#", "_").replace(":", "_").replace("-", "_")
    return f"{base_program_id}_opt_{safe_tl_id}_{schedule_time}"


def write_signal_override_xml(time_signal_plans, output_xml):
    root = ET.Element("additional", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
    })

    override_program_map = defaultdict(dict)

    for time_plan in time_signal_plans:
        schedule_time = int(round(float(time_plan["apply_time"])))
        for tl_id, plan in time_plan["signal_plan"].items():
            traffic_light = plan["traffic_light"]
            override_program_id = _make_override_program_id(
                traffic_light["programID"], tl_id, schedule_time
            )
            override_program_map[schedule_time][tl_id] = override_program_id

            tl_logic = ET.SubElement(root, "tlLogic", {
                "id": tl_id,
                "type": str(traffic_light["type"]),
                "programID": override_program_id,
                "offset": _format_num(float(traffic_light["offset"])),
            })

            for phase in plan["phases"]:
                phase_attrs = {
                    "duration": _format_num(float(phase["duration"])),
                    "state": str(phase["state"]),
                }
                if phase.get("hasMinDur"):
                    phase_attrs["minDur"] = _format_num(float(phase["minDur"]))
                if phase.get("hasMaxDur"):
                    phase_attrs["maxDur"] = _format_num(float(phase["maxDur"]))
                ET.SubElement(tl_logic, "phase", phase_attrs)

    ET.ElementTree(root).write(output_xml, encoding="utf-8", xml_declaration=True)
    return dict(override_program_map)


def write_empty_override_xml(output_xml):
    root = ET.Element("additional", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
    })
    ET.ElementTree(root).write(output_xml, encoding="utf-8", xml_declaration=True)
    return {}


def select_strategy_from_prediction(_prediction_csv):
    ranked_rows = [
        {
            "strategy": "no_control",
            "no_control": True,
            "predicted_score": 0.0,
        },
        {
            "strategy": "baseline_original",
            "is_proportional": True,
            "proportional_pool": 6.0,
            "top_n_tls": 3,
            "use_downstream_penalty": True,
            "update_interval": 180.0,
            "top_edge_count_override": 12,
            "predicted_score": 0.0,
        },
        {
            "strategy": "baseline_more_edges",
            "is_proportional": True,
            "proportional_pool": 6.0,
            "top_n_tls": 3,
            "use_downstream_penalty": True,
            "update_interval": 180.0,
            "top_edge_count_override": 20,
            "predicted_score": 0.0,
        },
        {
            "strategy": "baseline_more_edges_more_tls",
            "is_proportional": True,
            "proportional_pool": 8.0,
            "top_n_tls": 4,
            "use_downstream_penalty": True,
            "update_interval": 180.0,
            "top_edge_count_override": 20,
            "predicted_score": 0.0,
        },
    ]
    profile = {}
    return ranked_rows, profile


def process_prediction_csv(prediction_csv, work_dir=None, run_simulations=True, strategy=None):
    prediction_csv = os.path.abspath(prediction_csv)
    source_stem = get_source_stem_from_prediction_csv(prediction_csv)
    strategy = resolve_strategy(strategy)
    work_dir = os.path.abspath(work_dir or os.path.join(DEFAULT_OUTPUT_ROOT, f"{source_stem}_predict"))
    os.makedirs(work_dir, exist_ok=True)

    local_prediction_csv = os.path.join(work_dir, os.path.basename(prediction_csv))
    if prediction_csv != local_prediction_csv:
        shutil.copy2(prediction_csv, local_prediction_csv)

    prediction_df = pd.read_csv(local_prediction_csv)
    no_control = bool(strategy.get("no_control", False))

    if no_control:
        time_signal_plans = []
        signal_plan_summary_df = pd.DataFrame(columns=["time", "apply_time", "tl_id", "score"])
        override_program_map = {}
    else:
        time_signal_plans, signal_plan_summary_df = get_signal_plan_from_prediction(
            prediction_df,
            strategy.get("top_n_tls", DEFAULT_TOP_N_TLS),
            strategy,
        )
        override_program_map = {}

    signal_plan_summary_csv = os.path.join(work_dir, f"{source_stem}_signal_plan_summary.csv")
    signal_plan_summary_df.to_csv(signal_plan_summary_csv, index=False)

    override_xml = os.path.join(work_dir, f"{source_stem}_signal_override.xml")
    if no_control:
        write_empty_override_xml(override_xml)
        override_program_map = {}
    else:
        override_program_map = write_signal_override_xml(time_signal_plans, override_xml)

    route_xml_src = find_route_xml_for_prediction(prediction_csv)
    route_xml_dst = os.path.join(work_dir, os.path.basename(route_xml_src))
    shutil.copy2(route_xml_src, route_xml_dst)

    filtered_route_xml = os.path.join(work_dir, f"{source_stem}_filtered.rou.xml")
    filter_route_file(route_xml_dst, filtered_route_xml, get_valid_edge_ids(NET_FILE))

    before_cfg = os.path.join(work_dir, f"{source_stem}_before.sumocfg")
    after_cfg = os.path.join(work_dir, f"{source_stem}_after.sumocfg")
    before_stats_xml = os.path.join(work_dir, f"{source_stem}_before_stats.xml")
    after_stats_xml = os.path.join(work_dir, f"{source_stem}_after_stats.xml")
    comparison_summary_csv = os.path.join(work_dir, f"{source_stem}_comparison_summary.csv")

    create_temp_sumo_cfg(
        filtered_route_xml,
        BASE_SUMOCFG,
        before_cfg,
        output_overrides={"statistic-output": before_stats_xml},
    )

    if no_control:
        create_temp_sumo_cfg(
            filtered_route_xml,
            BASE_SUMOCFG,
            after_cfg,
            output_overrides={"statistic-output": after_stats_xml},
        )
    else:
        create_temp_sumo_cfg(
            filtered_route_xml,
            BASE_SUMOCFG,
            after_cfg,
            additional_files=[override_xml],
            output_overrides={"statistic-output": after_stats_xml},
        )

    result = {
        "work_dir": work_dir,
        "prediction_csv": local_prediction_csv,
        "signal_plan_summary_csv": signal_plan_summary_csv,
        "route_xml": route_xml_dst,
        "filtered_route_xml": filtered_route_xml,
        "before_cfg": before_cfg,
        "after_cfg": after_cfg,
        "override_xml": override_xml,
        "comparison_summary_csv": comparison_summary_csv,
        "before_end_time": None,
        "after_end_time": None,
    }

    if not run_simulations:
        return result

    before_output_csv = os.path.join(work_dir, f"{source_stem}_before.csv")
    after_output_csv = os.path.join(work_dir, f"{source_stem}_after.csv")

    before_ok, before_end_time = run_sumo_simulation_with_end_time(before_cfg, before_output_csv)

    if no_control:
        after_ok, after_end_time = run_sumo_simulation_with_end_time(after_cfg, after_output_csv)
    else:
        after_ok, after_end_time = run_sumo_simulation_with_end_time(
            after_cfg,
            after_output_csv,
            override_program_map=override_program_map,
        )

    result["before_end_time"] = before_end_time if before_ok else None
    result["after_end_time"] = after_end_time if after_ok else None

    before_summary = summarize_stats_xml(before_stats_xml) if before_ok else {}
    after_summary = summarize_stats_xml(after_stats_xml) if after_ok else {}

    comparison_df = pd.DataFrame([
        {
            "metric": metric,
            "before": before_summary.get(metric),
            "after": after_summary.get(metric),
            "delta": (
                after_summary.get(metric) - before_summary.get(metric)
                if before_summary.get(metric) is not None and after_summary.get(metric) is not None
                else None
            ),
        }
        for metric in METRICS
    ])
    comparison_df.to_csv(comparison_summary_csv, index=False)
    result["comparison_df"] = comparison_df
    return result


def evaluate_strategy_worker(args):
    prediction_csv, base_work_dir, run_simulations, strategy = args
    strat_name = strategy["strategy"]
    strat_work_dir = os.path.join(base_work_dir, strat_name)

    global TO_EDGE_WEIGHT
    TO_EDGE_WEIGHT = 0.8 if strategy.get("use_downstream_penalty", False) else 0.0

    result = process_prediction_csv(
        prediction_csv,
        work_dir=strat_work_dir,
        run_simulations=run_simulations,
        strategy=strategy,
    )

    waiting_time_after = float("inf")
    end_time_after = float("inf")
    waiting_time_before = float("inf")
    end_time_before = float("inf")

    df_comp = result.get("comparison_df")
    if df_comp is None and os.path.exists(result["comparison_summary_csv"]):
        df_comp = pd.read_csv(result["comparison_summary_csv"])

    if df_comp is not None and not df_comp.empty:
        wait_row = df_comp[df_comp["metric"] == "avg_waiting_time"]
        if not wait_row.empty:
            waiting_time_after = float(wait_row.iloc[0]["after"])
            waiting_time_before = float(wait_row.iloc[0]["before"])

        end_row = df_comp[df_comp["metric"] == "simulation_end_time"]
        if not end_row.empty:
            end_time_after = float(end_row.iloc[0]["after"])
            end_time_before = float(end_row.iloc[0]["before"])

    strategy["actual_waiting_time"] = waiting_time_after
    strategy["actual_end_time"] = end_time_after
    strategy["baseline_waiting_time"] = waiting_time_before
    strategy["baseline_end_time"] = end_time_before
    return strategy, result


def run_prediction_driven_strategy(prediction_csv, work_dir=None, run_simulations=True):
    ranked_rows, profile = select_strategy_from_prediction(prediction_csv)
    source_stem = get_source_stem_from_prediction_csv(prediction_csv)

    base_work_dir = os.path.abspath(
        work_dir or os.path.join(DEFAULT_OUTPUT_ROOT, f"{source_stem}_predict_dynamic")
    )
    os.makedirs(base_work_dir, exist_ok=True)

    print(f"\n>>> 啟動多進程並行模擬測試 (共 {len(ranked_rows)} 個策略) <<<")

    args_list = [
        (prediction_csv, base_work_dir, run_simulations, strategy)
        for strategy in ranked_rows
    ]

    strategy_rows = []
    best_result = None
    best_waiting_time = float("inf")

    with ProcessPoolExecutor(max_workers=len(ranked_rows)) as executor:
        for strategy, result in executor.map(evaluate_strategy_worker, args_list):
            print(f"[{strategy['strategy']}] 評估完成！")
            strategy_rows.append(strategy)

            if strategy["actual_waiting_time"] < best_waiting_time:
                best_waiting_time = strategy["actual_waiting_time"]
                best_result = result

    for row in strategy_rows:
        row["selected"] = (row["actual_waiting_time"] == best_waiting_time)

    strategy_summary_csv = os.path.join(base_work_dir, f"{source_stem}_strategy_summary.csv")
    pd.DataFrame(strategy_rows).to_csv(strategy_summary_csv, index=False)

    print("\n================= 競賽結果總覽 =================")
    if strategy_rows:
        base_end = strategy_rows[0].get("baseline_end_time")
        base_wait = strategy_rows[0].get("baseline_waiting_time")
        print(f"【原始未優化 (Before)】 結束時間 = {base_end} 秒 | 等待時間 = {base_wait} 秒")
        print("----------------------------------------------")

    for row in strategy_rows:
        wait_diff = row.get("actual_waiting_time", 0) - row.get("baseline_waiting_time", 0)
        end_diff = row.get("actual_end_time", 0) - row.get("baseline_end_time", 0)
        wait_sign = "+" if wait_diff > 0 else ""
        end_sign = "+" if end_diff > 0 else ""
        mark = "🏆 (最佳)" if row.get("selected") else "   "

        print(
            f"{mark} {row['strategy']:<30} : "
            f"結束時間 = {row.get('actual_end_time')} 秒 ({end_sign}{end_diff:.2f}) | "
            f"等待時間 = {row.get('actual_waiting_time')} 秒 ({wait_sign}{wait_diff:.2f})"
        )

    print("==============================================")

    return {
        "base_work_dir": base_work_dir,
        "strategy_summary_csv": strategy_summary_csv,
        "selection_mode": "prediction_dynamic",
        "prediction_profile": profile,
        "best_result": best_result,
    }


def run_automated_strategy_search(*args, **kwargs):
    return run_prediction_driven_strategy(*args, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to prediction CSV")
    args = parser.parse_args()

    if args.csv:
        run_prediction_driven_strategy(args.csv)
    else:
        print("請提供預測 CSV 路徑，例如: python traffic_light_optimizer.py --csv output.csv")
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd


# ─── SUMO 號誌物理約束 ─────────────────────────────────────────────────────────
# 違反這些下限的 phase 會在 SUMO 模擬中導致非法相位切換或行人/車輛無安全通過
# 時間。預設值是道路工程的保守下限（FHWA MUTCD 與 SUMO 官方範例皆採用）。
# 若你的路口需更嚴格（例如有行人保護相位），可透過環境變數覆蓋。
MIN_YELLOW_DURATION = float(os.environ.get("TRAFFICVISION_MIN_YELLOW", "3.0"))
MIN_ALL_RED_DURATION = float(os.environ.get("TRAFFICVISION_MIN_ALL_RED", "1.0"))
MIN_GREEN_DURATION = float(os.environ.get("TRAFFICVISION_MIN_GREEN", "5.0"))


def classify_phase_kind(state):
    """根據 SUMO state 字串判斷 phase 種類。

    state 每個字元代表一個 controlled connection：
      r/s = red / red-stop
      y/Y = yellow（清空相位）
      g/G = green（小寫 g 為 yield-green）
      o/O = off / blinking
    判定優先序：yellow（含 'y'/'Y'）> all-red（全 r/s）> green（含 G/g）
    """
    if not state:
        return "unknown"
    s = str(state)
    if any(c in ("y", "Y") for c in s):
        return "yellow"
    if all(c in ("r", "s", "R", "S") for c in s):
        return "all_red"
    if any(c in ("g", "G") for c in s):
        return "green"
    return "other"


def _physics_min_duration(state):
    """回傳此 phase 應有的最低秒數（依其 kind）。"""
    kind = classify_phase_kind(state)
    if kind == "yellow":
        return MIN_YELLOW_DURATION
    if kind == "all_red":
        return MIN_ALL_RED_DURATION
    if kind == "green":
        return MIN_GREEN_DURATION
    return 0.0


def _format_num(value):
    if isinstance(value, int):
        return str(value)
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def extract_traffic_lights(tree):
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
            })

        traffic_lights.append({
            "id": tl_logic.get("id", "N/A"),
            "type": tl_logic.get("type", "N/A"),
            "programID": tl_logic.get("programID", "0"),
            "offset": float(tl_logic.get("offset", "0")),
            "phases": phases,
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
            "controlled_by_tl": False,
            "tl_id": None,
            "tl_link_index": None,
        })

    return connections


def link_connections_to_traffic_lights(connections, tree):
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
    """夾在 [min, max] 之間，同時不得低於 SUMO 物理下限（黃燈/全紅/綠燈）。

    黑箱下限（hasMinDur=False 時推算的 base*0.4）對黃燈這種固定 3s 的 phase
    來說可能算出 1.2s — 已違反道路工程下限。此處強制把物理下限疊加進去。
    """
    min_duration, max_duration = get_phase_duration_bounds(phase)
    physics_floor = _physics_min_duration(phase.get("state", ""))
    effective_min = max(min_duration, physics_floor)
    effective_max = max(effective_min, max_duration)
    return max(effective_min, min(effective_max, float(new_duration)))


def build_updated_phase(phase, duration):
    return {
        "duration": clamp_phase_duration(phase, duration),
        "state": phase["state"],
        "minDur": phase.get("minDur"),
        "maxDur": phase.get("maxDur"),
        "hasMinDur": phase.get("hasMinDur"),
        "hasMaxDur": phase.get("hasMaxDur"),
    }


def normalize_predicted_edge_scores(edge_scores, top_edge_count, min_edge_score):
    filtered = [
        (str(edge_id).strip("'"), float(score))
        for edge_id, score in edge_scores.items()
        if not str(edge_id).strip("'").startswith(":") and float(score) >= min_edge_score
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


def build_tls_scores_from_predictions(
    connections,
    edge_scores,
    strategy,
    default_top_edge_count,
    min_edge_score,
    from_edge_weight,
    to_edge_weight,
    unsafe_tls_ids,
):
    tls_scores = defaultdict(float)
    tls_links = defaultdict(set)
    link_scores = defaultdict(float)

    top_edge_count = int(strategy.get("top_edge_count_override", default_top_edge_count))
    normalized_edge_scores = normalize_predicted_edge_scores(edge_scores, top_edge_count, min_edge_score)

    if not normalized_edge_scores:
        return tls_scores, tls_links, link_scores

    for conn in connections:
        if not conn.get("controlled_by_tl") or conn["tl_id"] in unsafe_tls_ids:
            continue

        from_edge = conn["from_edge"].strip("'")
        to_edge = conn["to_edge"].strip("'")
        score = (
            normalized_edge_scores.get(from_edge, 0.0) * from_edge_weight
            - normalized_edge_scores.get(to_edge, 0.0) * to_edge_weight
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


def build_signal_plan_from_edge_scores(
    traffic_lights,
    connections,
    edge_scores,
    top_n_tls,
    strategy,
    default_top_edge_count,
    min_edge_score,
    from_edge_weight,
    to_edge_weight,
    unsafe_tls_ids,
    top_phases_to_adjust,
    primary_phase_extra_seconds,
    secondary_phase_extra_seconds,
):
    traffic_light_map = {tl["id"]: tl for tl in traffic_lights}

    tls_scores, tls_links, link_scores = build_tls_scores_from_predictions(
        connections,
        edge_scores,
        strategy,
        default_top_edge_count,
        min_edge_score,
        from_edge_weight,
        to_edge_weight,
        unsafe_tls_ids,
    )

    selected_tl_ids = [
        tl_id
        for tl_id, _ in sorted(tls_scores.items(), key=lambda item: item[1], reverse=True)[:int(strategy.get("top_n_tls", top_n_tls))]
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
                if total_phase_score > 0 and ("G" in phase["state"] or "g" in phase["state"]) and phase_score_map.get(phase_index, 0) > 0:
                    new_duration += (phase_score_map[phase_index] / total_phase_score) * proportional_pool
                updated_phases.append(build_updated_phase(phase, new_duration))

        elif strategy.get("preserve_cycle", False) and len(phase_scores) >= 2:
            busiest_phase = phase_scores[0][0]
            least_busy_phase = phase_scores[-1][0]
            primary_extra = max(0.0, float(strategy.get("primary_phase_extra", primary_phase_extra_seconds)))

            for phase_index, phase in enumerate(tl["phases"]):
                delta = primary_extra if phase_index == busiest_phase else -primary_extra if phase_index == least_busy_phase else 0.0
                updated_phases.append(build_updated_phase(phase, float(phase["duration"]) + delta))

        else:
            top_count = max(1, int(strategy.get("top_phases_to_adjust", top_phases_to_adjust)))
            selected_phase_indices = {phase_index for phase_index, _ in phase_scores[:top_count]}
            best_phase = phase_scores[0][0]
            primary_extra = max(0.0, float(strategy.get("primary_phase_extra", primary_phase_extra_seconds)))
            secondary_extra = max(0.0, float(strategy.get("secondary_phase_extra", secondary_phase_extra_seconds)))

            for phase_index, phase in enumerate(tl["phases"]):
                if phase_index == best_phase and phase_index in selected_phase_indices:
                    delta = primary_extra
                elif phase_index in selected_phase_indices:
                    delta = secondary_extra
                else:
                    delta = 0.0
                updated_phases.append(build_updated_phase(phase, float(phase["duration"]) + delta))

        signal_plan[tl_id] = {"traffic_light": tl, "phases": updated_phases}
        selected_summary.append((tl_id, tls_scores[tl_id]))

    return signal_plan, selected_summary


def build_tls_road_hint_map(connections):
    tls_roads = defaultdict(set)
    for conn in connections:
        if not conn.get("controlled_by_tl"):
            continue
        tl_id = conn.get("tl_id")
        if not tl_id:
            continue
        from_name = str(conn.get("from_edge_name") or "").strip()
        to_name = str(conn.get("to_edge_name") or "").strip()
        from_edge = str(conn.get("from_edge") or "").strip("'")
        to_edge = str(conn.get("to_edge") or "").strip("'")

        tls_roads[tl_id].add(from_name if from_name else from_edge)
        tls_roads[tl_id].add(to_name if to_name else to_edge)

    return {
        tl_id: " | ".join(sorted(name for name in names if name))
        for tl_id, names in tls_roads.items()
    }


def get_signal_plan_from_prediction(
    prediction_df,
    top_n_tls,
    strategy,
    net_file,
    default_top_edge_count,
    min_edge_score,
    from_edge_weight,
    to_edge_weight,
    unsafe_tls_ids,
    top_phases_to_adjust,
    primary_phase_extra_seconds,
    secondary_phase_extra_seconds,
):
    tree = ET.parse(net_file)
    traffic_lights = extract_traffic_lights(tree)
    connections = extract_connections(tree, exclude_internal=True)
    connections = link_connections_to_traffic_lights(connections, tree)

    base_prediction_time = float(prediction_df["time"].min())
    update_interval = float(strategy.get("update_interval", 120.0))

    prediction_df = prediction_df.copy()
    prediction_df["time_bin"] = (
        ((prediction_df["time"] - base_prediction_time) // update_interval) * update_interval
        + base_prediction_time
    )

    time_signal_plans = []
    summary_rows = []
    tls_road_hint_map = build_tls_road_hint_map(connections)

    for current_bin, time_df in prediction_df.groupby("time_bin", sort=True):
        apply_time = max(0.0, float(current_bin) - base_prediction_time)
        edge_scores = defaultdict(float)

        for row in time_df.itertuples(index=False):
            edge_scores[str(row.edge_id)] += float(row.vehicle_count)

        signal_plan, selected_summary = build_signal_plan_from_edge_scores(
            traffic_lights,
            connections,
            edge_scores,
            top_n_tls,
            strategy,
            default_top_edge_count,
            min_edge_score,
            from_edge_weight,
            to_edge_weight,
            unsafe_tls_ids,
            top_phases_to_adjust,
            primary_phase_extra_seconds,
            secondary_phase_extra_seconds,
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

    return time_signal_plans, pd.DataFrame(summary_rows), tls_road_hint_map


def build_signal_change_detail_df(time_signal_plans, tls_road_hint_map=None):
    tls_road_hint_map = tls_road_hint_map or {}
    rows = []

    for time_plan in time_signal_plans:
        schedule_time = float(time_plan.get("time", 0.0))
        apply_time = float(time_plan.get("apply_time", 0.0))

        for tl_id, plan in time_plan.get("signal_plan", {}).items():
            original_phases = plan["traffic_light"]["phases"]
            updated_phases = plan["phases"]

            for phase_index, (orig_phase, updated_phase) in enumerate(zip(original_phases, updated_phases)):
                original_duration = float(orig_phase.get("duration", 0.0))
                updated_duration = float(updated_phase.get("duration", 0.0))
                delta_duration = updated_duration - original_duration

                if abs(delta_duration) < 1e-9:
                    continue

                rows.append({
                    "time": schedule_time,
                    "apply_time": apply_time,
                    "tl_id": tl_id,
                    "road_hint": tls_road_hint_map.get(tl_id, ""),
                    "phase_index": phase_index,
                    "state": str(updated_phase.get("state", "")),
                    "old_duration": original_duration,
                    "new_duration": updated_duration,
                    "delta_duration": delta_duration,
                })

    columns = [
        "time",
        "apply_time",
        "tl_id",
        "road_hint",
        "phase_index",
        "state",
        "old_duration",
        "new_duration",
        "delta_duration",
    ]
    return pd.DataFrame(rows, columns=columns)


def make_override_program_id(base_program_id, tl_id, schedule_time):
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
            override_program_id = make_override_program_id(
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

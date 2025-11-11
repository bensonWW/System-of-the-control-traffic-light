import os
import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Dict

import grabapi
import searchnetdata as sd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ROUTE_FILE = os.path.join(DATA_DIR, "output.rou.xml")
ALT_ROUTE_FILE = os.path.join(DATA_DIR, "output.rou.alt.xml")
FLOW_TEMPLATE_FILE = os.path.join(DATA_DIR, "generated_flows.rou.xml")
NET_FILE = os.path.join(DATA_DIR, "ntut-the way.net.xml")

XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ROUTE_SCHEMA = "http://sumo.dlr.de/xsd/routes_file.xsd"

FLOW_PREFIX = "api_flow_"
VTYPE_PREFIX = "api_vt_"

SIMULATION_END = 3600  # seconds
SPEED_BUFFER_RATIO = 0.8
DEFAULT_MAX_SPEED_KMH = 40.0
MIN_MAX_SPEED_MPS = 1.0
ET.register_namespace("xsi", XSI_NS)

def _create_routes_root() -> ET.Element:
    root = ET.Element("routes")
    root.set(f"{{{XSI_NS}}}noNamespaceSchemaLocation", ROUTE_SCHEMA)
    return root

def _is_in_scope(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
) -> bool:
    start_inside = lon_min <= start_lon <= lon_max and lat_min <= start_lat <= lat_max
    end_inside = lon_min <= end_lon <= lon_max and lat_min <= end_lat <= lat_max
    return start_inside or end_inside


def _normalise(value: float, lower: float, upper: float, delta: float) -> float:
    return ((value - lower) / (upper - lower)) * delta

def fetch_and_prepare_road_data() -> Dict[str, Dict[str, Any]]:
    raw_data = grabapi.getData()
    lon_min, lat_min, lon_max, lat_max = sd.getmapscope()
    conv_boundary = list(map(float, sd.getConverposition()))
    conv_delta_x = conv_boundary[2]
    conv_delta_y = conv_boundary[3]

    prepared: Dict[str, Dict[str, Any]] = {}
    for road_name, attributes in raw_data.items():
        try:
            start_lon = float(attributes["StartWgsX"])
            start_lat = float(attributes["StartWgsY"])
            end_lon = float(attributes["EndWgsX"])
            end_lat = float(attributes["EndWgsY"])
        except (KeyError, TypeError, ValueError):
            continue

        if not _is_in_scope(
            lon_min, lat_min, lon_max, lat_max, start_lon, start_lat, end_lon, end_lat
        ):
            continue

        converted = attributes.copy()
        converted["StartWgsX"] = _normalise(start_lon, lon_min, lon_max, conv_delta_x)
        converted["StartWgsY"] = _normalise(start_lat, lat_min, lat_max, conv_delta_y)
        converted["EndWgsX"] = _normalise(end_lon, lon_min, lon_max, conv_delta_x)
        converted["EndWgsY"] = _normalise(end_lat, lat_min, lat_max, conv_delta_y)

        try:
            converted["from"] = sd.search(
                converted["StartWgsX"], converted["StartWgsY"]
            )
            converted["to"] = sd.search(converted["EndWgsX"], converted["EndWgsY"])
        except ValueError:
            continue

        prepared[road_name] = converted

    print(f"Prepared {len(prepared)} road segment(s) within the network boundary.")
    return prepared

def _sanitize_identifier(value: str, fallback: str) -> str:
    if not value:
        value = fallback
    sanitized = []
    for ch in value:
        if ch.isalnum():
            sanitized.append(ch.lower())
        else:
            sanitized.append("_")
    cleaned = "".join(sanitized).strip("_")
    return cleaned or fallback


def _compute_max_speed(avg_speed_kmh: Any) -> float:
    try:
        avg_speed = float(avg_speed_kmh)
    except (TypeError, ValueError):
        avg_speed = 0.0

    if avg_speed <= 0.0:
        return DEFAULT_MAX_SPEED_KMH / 3.6

    boosted = avg_speed / SPEED_BUFFER_RATIO / 3.6
    return max(boosted, MIN_MAX_SPEED_MPS)


def _parse_volume(total_volume: Any) -> int:
    try:
        volume = int(float(total_volume))
    except (TypeError, ValueError):
        return 0
    return max(volume, 0)

def _indent_tree(tree: ET.ElementTree) -> None:
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        _legacy_indent(tree.getroot())


def _legacy_indent(elem: ET.Element, level: int = 0) -> None:
    indent = "    "
    newline = "\n"
    children = list(elem)
    if children:
        if not elem.text or not elem.text.strip():
            elem.text = newline + indent * (level + 1)
        for child in children:
            _legacy_indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = newline + indent * level
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = newline + indent * level

def import_traffic_into_routes(root: ET.Element, prepared_data: Dict[str, Dict[str, Any]]) -> int:
    inserted = 0
    declared_vtypes = set()

    for index, (road_name, data) in enumerate(prepared_data.items()):
        total_volume = _parse_volume(data.get("TotalVol"))
        if total_volume <= 0:
            continue

        from_junction = data.get("from")
        to_junction = data.get("to")
        if not from_junction or not to_junction:
            continue

        section_id = data.get("SectionId")
        safe_section = _sanitize_identifier(section_id or str(index), f"s{index}")

        flow_id = f"{FLOW_PREFIX}{safe_section}"
        vtype_id = f"{VTYPE_PREFIX}{safe_section}"
        max_speed = _compute_max_speed(data.get("AvgSpd"))

        if vtype_id not in declared_vtypes:
            vtype_attrs = {
                "id": vtype_id,
                "vClass": "passenger",
                "length": "5.0",
                "color": "0,1,0",
                "maxSpeed": f"{max_speed:.2f}",
            }
            ET.SubElement(root, "vType", attrib=vtype_attrs)
            declared_vtypes.add(vtype_id)

        flow_attrs = {
            "id": flow_id,
            "type": vtype_id,
            "begin": "0",
            "end": str(SIMULATION_END),
            "number": str(total_volume),
            "fromJunction": from_junction,
            "toJunction": to_junction,
            "departLane": "best",
            "departPos": "random_free",
            "departSpeed": "max",
        }
        flow_elem = ET.SubElement(root, "flow", attrib=flow_attrs)
        ET.SubElement(
            flow_elem, "param", attrib={"key": "RoadName", "value": road_name}
        )
        if data.get("SectionId"):
            ET.SubElement(
                flow_elem, "param", attrib={"key": "SectionId", "value": data["SectionId"]}
            )
        if data.get("AvgSpd"):
            ET.SubElement(
                flow_elem, "param", attrib={"key": "AverageSpeedKmh", "value": data["AvgSpd"]}
            )
        inserted += 1

    print(f"Inserted {inserted} flow(s) derived from live traffic volume.")
    return inserted


def build_flow_template(prepared_data: Dict[str, Dict[str, Any]], template_path: str) -> int:
    root = _create_routes_root()
    inserted = import_traffic_into_routes(root, prepared_data)
    tree = ET.ElementTree(root)
    _indent_tree(tree)
    tree.write(template_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote flow template to '{template_path}'.")
    return inserted


def run_duarouter(flow_file: str, output_file: str, alt_file: str) -> None:
    if not os.path.exists(flow_file):
        raise FileNotFoundError(f"Flow template '{flow_file}' not found.")

    cmd = [
        "duarouter",
        "-n",
        NET_FILE,
        "-r",
        flow_file,
        "-o",
        output_file,
        "--alternatives-output",
        alt_file,
        "--max-alternatives",
        "3",
        "--weights.random-factor",
        "1.75",
        "--junction-taz",
        "--ignore-errors",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    print("duarouter STDOUT:\n", result.stdout)
    print("duarouter STDERR:\n", result.stderr)
    result.check_returncode()


def beautify_route_file(route_path: str) -> None:
    if not os.path.exists(route_path):
        return
    tree = ET.parse(route_path)
    _indent_tree(tree)
    tree.write(route_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    prepared_data = fetch_and_prepare_road_data()
    inserted = build_flow_template(prepared_data, FLOW_TEMPLATE_FILE)
    if inserted == 0:
        print("No flows were added. Check if the API returned usable data.")
        return
    run_duarouter(FLOW_TEMPLATE_FILE, ROUTE_FILE, ALT_ROUTE_FILE)
    beautify_route_file(ROUTE_FILE)
    print(f"Route file updated at '{ROUTE_FILE}'.")

if __name__ == "__main__":
    main()

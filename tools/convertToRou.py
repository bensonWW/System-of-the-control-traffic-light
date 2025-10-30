import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, Tuple

import grabapi
import searchnetdata as sd

"""
Line 1 ~ 6
匯入標準模組 os、xml.etree.ElementTree、typing，以及專案內的 grabapi 與 searchnetdata，前者抓取即時交通資料，後者提供 SUMO 網路的座標與節點查詢。
"""

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ROUTE_FILE = os.path.join(DATA_DIR, "output.rou.xml")

XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ROUTE_SCHEMA = "http://sumo.dlr.de/xsd/routes_file.xsd"

FLOW_PREFIX = "api_flow_"
VTYPE_PREFIX = "api_vt_"

SIMULATION_END = 3600  # seconds
SPEED_BUFFER_RATIO = 0.8
DEFAULT_MAX_SPEED_KMH = 40.0
MIN_MAX_SPEED_MPS = 1.0

"""
Line 13 ~ 26
設定路徑常數（專案根目錄、資料夾、output.rou.xml）、XML namespace、流程和車種 ID 前綴，以及模擬結束時間、速度緩衝係數、預設/最低速度，供後續統一引用。
"""

def inspect_route_file(path: str) -> Tuple[ET.ElementTree, ET.Element]:
    if not os.path.exists(path):
        root = _create_routes_root()
        tree = ET.ElementTree(root)
        print(f"Route file '{path}' not found. Creating a new routes container.")
        return tree, root

    tree = ET.parse(path)
    root = tree.getroot()
    flow_count = len(root.findall("flow"))
    vehicle_count = len(root.findall("vehicle"))
    print(
        f"Route file '{path}' currently contains {flow_count} flow(s) and "
        f"{vehicle_count} vehicle(s)."
    )
    return tree, root

def _create_routes_root() -> ET.Element:
    root = ET.Element("routes")
    root.set("xmlns:xsi", XSI_NS)
    root.set(f"{{{XSI_NS}}}noNamespaceSchemaLocation", ROUTE_SCHEMA)
    return root

"""
Line 33 ~ 54
inspect_route_file 會載入現有的 output.rou.xml，並回報目前有多少 <flow> 與 <vehicle>；
若檔案不存在則透過 _create_routes_root (tools/convertToRou.py (lines 46-51)) 建立帶有 schema 定義的空 <routes> 元素。
"""

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

"""
Line 62 ~ 78
_is_in_scope 判斷路段是否落在地圖邊界，_normalise 依據網路座標範圍把經緯度換算成 SUMO 的平面座標。
"""

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

"""
Line 85 ~ 124
fetch_and_prepare_road_data 透過 grabapi.getData() 拿到 raw 資料，利用 searchnetdata 提供的地圖邊界與座標轉換，把每條路段的起訖點換算成 SUMO 座標，再透過 sd.search 找出最近的交通號誌節點；
篩掉不在範圍內或資料缺漏的路段後回傳整理好的字典。
"""

def _clean_previous_entries(root: ET.Element) -> None:
    for flow in list(root.findall("flow")):
        if flow.get("id", "").startswith(FLOW_PREFIX):
            root.remove(flow)
    for vtype in list(root.findall("vType")):
        if vtype.get("id", "").startswith(VTYPE_PREFIX):
            root.remove(vtype)

"""
Line 132 ~ 138
_clean_previous_entries 將既有由本程式產生（ID 以 api_ 前綴）的 <flow> 與 <vType> 移除，避免重複累積。
"""

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

"""
Line 145 ~ 176
_sanitize_identifier 把 SectionId 等字串轉成 SUMO 合法的 ID；
_compute_max_speed 依據平均車速加上緩衝（除以 SPEED_BUFFER_RATIO，換算成 m/s），低於等於 0 時使用預設速度；
_parse_volume 解析 TotalVol 並確保不為負值。
"""

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

"""
Line 185 ~ 205
_indent_tree 嘗試使用 Python 3.9+ 的 ET.indent 排版輸出，若環境較舊則改用 _legacy_indent 逐層補上換行與縮排。
"""

def import_traffic_into_routes(root: ET.Element, prepared_data: Dict[str, Dict[str, Any]]) -> int:
    _clean_previous_entries(root)
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

"""
Line 212 ~ 272
import_traffic_into_routes 先清除舊資料，再對每條合法路段建立一組 <vType>（若尚未產生）與 <flow>：
    依 SectionId 生成穩定的 flow/vType ID。
    設定流量起訖時間、車輛數、起終端節點與出發策略。
    用 <param> 儲存道路名稱、SectionId、平均車速，避免使用 schema 不允許的 name 屬性。
"""

def main() -> None:
    tree, root = inspect_route_file(ROUTE_FILE)
    prepared_data = fetch_and_prepare_road_data()
    inserted = import_traffic_into_routes(root, prepared_data)
    if inserted == 0:
        print("No flows were added. Check if the API returned usable data.")
    _indent_tree(tree)
    tree.write(ROUTE_FILE, encoding="utf-8", xml_declaration=True)
    print(f"Route file updated at '{ROUTE_FILE}'.")

"""
Line 282 ~ 290
main() 串起整個流程：載入或建立路徑檔、整理 API 資料、寫入新的 <flow> 與 <vType>、排版後輸出 XML，最後在主程式入口點執行。
"""

if __name__ == "__main__":
    main()

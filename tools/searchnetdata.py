import xml.etree.ElementTree as ET
import math
def getRoot():
    path = "./data/ntut_network_split.net copy.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    return root
def getConverposition():
    return getRoot()[0].attrib["convBoundary"].split(",")
def getmapscope():
    root = getRoot()
    mapInformation = root[0].attrib["origBoundary"]
    lon_min = float(mapInformation.split(",")[0])  #最左邊的經度
    lat_min = float(mapInformation.split(",")[1]) #最下面的緯度
    lon_max = float(mapInformation.split(",")[2]) #最右邊的經度
    lat_max = float(mapInformation.split(",")[3]) #最上面的緯度
    return (lon_min,lat_min,lon_max,lat_max)
def search(position_x, position_y):
    root = getRoot()
    roadInfo = {}
    position_x, position_y = float(position_x), float(position_y)

    for child in root:
        # Search ALL non-internal junctions (not just traffic_light)
        # to handle merged clusters correctly
        if child.tag == "junction" and not child.attrib["id"].startswith(":"):
            jtype = child.attrib.get("type", "")
            if jtype in ("traffic_light", "priority"):
                x, y = float(child.attrib["x"]), float(child.attrib["y"])
                dist = math.dist((x, y), (position_x, position_y))
                roadInfo[child.attrib["id"]] = [x, y, dist]
    minid = min(roadInfo, key=lambda k: roadInfo[k][2])
    return minid


# Cache for junction remapping
_all_junction_ids = None

def _get_all_junction_ids():
    """Load and cache all non-internal junction IDs from the network file."""
    global _all_junction_ids
    if _all_junction_ids is None:
        root = getRoot()
        _all_junction_ids = set()
        for child in root:
            if child.tag == "junction" and not child.attrib["id"].startswith(":"):
                _all_junction_ids.add(child.attrib["id"])
    return _all_junction_ids


def remap_junction(jid):
    """
    Remap a junction ID to the correct one in the current .net.xml.
    If jid exists, return as-is. Otherwise find a cluster that contains jid
    as a substring (handles merged junctions like 619136881 -> cluster619136880_619136881_...).
    """
    all_ids = _get_all_junction_ids()
    if jid in all_ids:
        return jid
    # Try substring match: find a cluster that contains this junction ID
    candidates = [nid for nid in all_ids if jid in nid]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Multiple matches — pick the shortest (most specific)
        return min(candidates, key=len)
    # No match found, return original (duarouter will warn)
    return jid
import xml.etree.ElementTree as ET
import math
def getRoot():
    path = "./data/ntut-the way.net.xml"
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
        if child.tag == "junction" and not child.attrib["id"].startswith(":") and child.attrib["type"] == "traffic_light":
            x, y = float(child.attrib["x"]), float(child.attrib["y"])
            dist = math.dist((x, y), (position_x, position_y))
            roadInfo[child.attrib["id"]] = [x, y, dist]
    minid = min(roadInfo, key=lambda k: roadInfo[k][2])
    return minid
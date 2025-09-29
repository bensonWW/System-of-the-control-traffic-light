import xml.etree.ElementTree as ET
def getRoot():
    path = "./data/ntut-the way.net.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    return root
def getmapscope():
    root = getRoot()
    mapInformation = root[0].attrib["origBoundary"]
    lon_min = float(mapInformation.split(",")[0])  #最左邊的經度
    lat_min = float(mapInformation.split(",")[1]) #最下面的緯度
    lon_max = float(mapInformation.split(",")[2]) #最右邊的經度
    lat_max = float(mapInformation.split(",")[3]) #最上面的緯度
    return (lon_min,lat_min,lon_max,lat_max)
def search(road1,road2,road3):
    root = getRoot()
    roadInfoNametype = {}
    for child in root:
        if child.tag == "edge" and "function" not in child.attrib and "name" in child.attrib:
            name = child.attrib["name"]
            fromm = child.attrib["from"]
            to = child.attrib["to"]
            idd = child.attrib["id"]
            if name not in roadInfoNametype:
                roadInfoNametype[name] = {}
            if idd not in roadInfoNametype[name]:
                roadInfoNametype[name][idd] = {}
            roadInfoNametype[name][idd]["from"] = fromm
            roadInfoNametype[name][idd]["to"] = to
    if road1 not in roadInfoNametype or road2 not in roadInfoNametype or road3 not in roadInfoNametype:
        return
    road1List = []
    fromm = ""
    to = ""
    for idd in roadInfoNametype[road1]:
        fromm = roadInfoNametype[road1][idd]["from"]
        to = roadInfoNametype[road1][idd]["to"]
        road1List.append([fromm,to])
    for idd in roadInfoNametype[road2]:
        for FT in road1List:
            if roadInfoNametype[road2][idd]["from"] in FT:
                fromm = roadInfoNametype[road2][idd]["from"]
            elif roadInfoNametype[road2][idd]["to"] in FT:
                fromm = roadInfoNametype[road2][idd]["to"]
    for idd in roadInfoNametype[road3]:
        for FT in road1List:
            if roadInfoNametype[road3][idd]["from"] in FT:
                to = roadInfoNametype[road3][idd]["from"]
            elif roadInfoNametype[road3][idd]["to"] in FT:
                to = roadInfoNametype[road3][idd]["to"]
    return[fromm,to]
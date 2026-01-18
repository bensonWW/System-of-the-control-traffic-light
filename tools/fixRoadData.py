import xml.etree.ElementTree as ET
import sumolib as SLB
def getMapData():
    path = "./data/ntut-the way.net.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    temp = {}
    for child in root:
        if(child.tag == "edge"):
            if("function" not in child.attrib):
                temp[child.attrib["id"]] = [child.attrib["from"], child.attrib["to"], False]
    return temp
def markMapData(mapData):
    path = "./data/output.rou.xml"
    tree = ET.parse(path)
    tripInfo = tree.getroot()
    for edgeId in mapData:
        for info in tripInfo:
            if(info.tag == "vehicle"):
                route = info.find("route")
                edges = route.attrib["edges"].split(" ")
                for e in edges:
                    if(e == edgeId):
                        mapData[edgeId][2] = True
                        break
    return mapData
def fixtheRoadData():
    mapData = markMapData(getMapData())
    path = "./data/ntut-the way.net.xml"
    tripPath = "./data/output.rou.xml"
    root = ET.parse(tripPath).getroot()
    net = SLB.net.readNet(path)
    for edgeId in mapData:
        vol = 0
        if(mapData[edgeId][2] == False):
            continue
            # vol += 
if __name__ == "__main__":
    print(fixtheRoadData())
import xml.etree.ElementTree as ET
import sumolib as SLB
import selectRoad as ST
def getMapData():
    path = "./data/ntut-the way.net.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    temp = {}
    for child in root:
        if(child.tag == "edge"):
            if("function" not in child.attrib):
                temp[child.attrib["id"]] = [child.attrib["from"], child.attrib["to"]]
    return temp
def getEdgesVolume():
    finalInfo = {}
    path = "./data/output.rou.xml"
    tripRoot = ET.parse(path).getroot()
    roadInfo = ST.select() #'八德路    松江路-忠孝東路': {'SectionId': 'ZJTJ960', 'AvgSpd': '45.435482', 'AvgOcc': '1.6', 'TotalVol': '62.0', 'MOELevel': '0', 'StartWgsX': 392.3516351377892, 'StartWgsY': 366.40778006647395, 'EndWgsX': 105.22111980907044, 'EndWgsY': 285.28874187925237, 'from': 'cluster_2528018119_655375236', 'to': '655375228'}
    for info in tripRoot:
        if(info.tag == "vehicle"):
                fromNode = info.attrib["fromTaz"]
                toNode = info.attrib["toTaz"]
                passRoads = info.find("route").attrib["edges"].split(" ")
        else:
            continue
        for name in roadInfo:
            if(roadInfo[name]["from"] == fromNode and roadInfo[name]["to"] == toNode):
                passNum = 0
                for edgeId in passRoads:
                    if edgeId not in finalInfo:
                        finalInfo[edgeId] = [int(float(roadInfo[name]["TotalVol"])) * pow(0.75, passNum),1]
                    else:
                        finalInfo[edgeId][0] += int(float(roadInfo[name]["TotalVol"])) * pow(0.75, passNum)
                        finalInfo[edgeId][1] += 1
                    passNum += 1
    for edgeId in finalInfo:
        finalInfo[edgeId] = finalInfo[edgeId][0] // finalInfo[edgeId][1]
    return finalInfo
def generateEmptyEdgesVolume(net,edgesVolume,roadId,visited):
    ALPHA = 0.35
    inComingedges = net.getEdge(roadId).getFromNode().getIncoming()
    vol = 0
    for inEdge in inComingedges:
        inEdgeId = inEdge.getID()
        if inEdgeId in visited:
            continue
        else:
            visited.add(inEdgeId)
        if edgesVolume[inEdgeId] == 0:
            vol += generateEmptyEdgesVolume(net,edgesVolume,inEdgeId,visited) * ALPHA
        else:
            vol += edgesVolume[inEdgeId] * ALPHA
    return vol

def fixtheRoadData():
    net_path = "./data/ntut-the way.net.xml"
    mapData = getMapData()
    edgesVolume = getEdgesVolume()
    net = SLB.net.readNet(net_path)
    for edgeId in mapData:
        if edgeId not in edgesVolume:
            edgesVolume[edgeId] = 0
    for edgeId in edgesVolume:
        if edgesVolume[edgeId] == 0:
            edgesVolume[edgeId] += round(generateEmptyEdgesVolume(net,edgesVolume,edgeId,set()))
    temp = getEdgesVolume()
    for edgeId in temp:
        del edgesVolume[edgeId]
    return edgesVolume
def addFlow(edgesVolume,T = 300):
    tree = ET.parse("./data/output.rou.alt.xml")
    root = tree.getroot()
    edges = []
    for edge, num in edgesVolume.items():
        if num > 0:
            edges += [edge] * int(num)
    if not edges:
        return
    gap = T / len(edges)
    depart = 0.0
    for i, edge in enumerate(edges):
        v = ET.SubElement(root, "vehicle", {
            "id": f"addVeh_{i}",
            "type": "Car2",
            "depart": f"{depart:.2f}"
        })
        ET.SubElement(v, "route", {"edges": edge})
        depart += gap
    vehicles = [c for c in root if c.tag == "vehicle"]
    others = [c for c in root if c.tag != "vehicle"]
    vehicles.sort(key=lambda x: float(x.get("depart", 0)))
    root[:] = others + vehicles
    tree.write("./data/output.rou.alt.xml", encoding="utf-8", xml_declaration=True)
if __name__ == "__main__":
    edgesVolume = fixtheRoadData()
    print(edgesVolume)
    for edgeId in edgesVolume:
        if edgesVolume[edgeId] == 0:
            print(edgeId)
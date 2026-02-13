import xml.etree.ElementTree as ET
from math import floor
import sumolib as SLB
import selectRoad as ST
import searchnetdata as SD
def getMapData():
    path = "./data/ntut_network_split.net copy.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    temp = {}
    for child in root:
        if(child.tag == "edge"):
            if("function" not in child.attrib):
                temp[child.attrib["id"]] = [child.attrib["from"], child.attrib["to"]]
    return temp
def getEdgesVolume(road_info=None):
    finalInfo = {}
    path = "./data/output.rou.xml"
    tripRoot = ET.parse(path).getroot()
    if road_info is None:
        roadInfo = ST.select()
        # Remap stale junction IDs to current .net.xml cluster names
        for name in roadInfo:
            roadInfo[name]["from"] = SD.remap_junction(roadInfo[name]["from"])
            roadInfo[name]["to"] = SD.remap_junction(roadInfo[name]["to"])
    else:
        roadInfo = road_info
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
#利用遞迴補足缺失edge的車流資訊
def generateEmptyEdgesVolume(net,edgesVolume,roadId,visited):
    ALPHA = 0.25
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
#深層搜尋
def deepSearch(id,net,trips,edgesVolume,Edges,remainingEdges):
    LIMITPASSEDGE = 12
    if len(Edges) == 0:
        trips["trip:" + id]["to"] = net.getEdge(trips["trip:" + id]["pass"][-1]).getToNode().getID()
        return trips
    for edge in Edges:
        edgeId = edge.getID()
        if edgeId in remainingEdges and len(trips["trip:" + id]["pass"]) < LIMITPASSEDGE:
            Edges = net.getEdge(edgeId).getToNode().getOutgoing()
            trips["trip:" + id]["pass"].append(edgeId)
            remainingEdges.remove(edgeId)
            trips = deepSearch(id,net,trips,edgesVolume,Edges,remainingEdges)
            break
        trips["trip:" + id]["to"] = net.getEdge(trips["trip:" + id]["pass"][-1]).getToNode().getID()
    return trips
#生成trips
def findTrip(net,edgesVolume):
    trips = {}
    tripId = 1
    remainingEdges = set(edgesVolume.keys())
    while len(remainingEdges) > 0:
        id = str(tripId)
        trips["trip:" + id] = {"pass" : [] , "from": "", "to": "", "TotalVol" : 0}
        startEdge = next(iter(remainingEdges))
        trips["trip:" + id]["pass"].append(startEdge)
        trips["trip:" + id]["from"] = net.getEdge(startEdge).getFromNode().getID()
        remainingEdges.remove(startEdge)
        outComingEdges = net.getEdge(startEdge).getToNode().getOutgoing()
        trips = deepSearch(id,net,trips,edgesVolume,outComingEdges,remainingEdges)
        tripId += 1
    return trips
#完成每個trip的車流
def completeTheVol(trips,edgesVolume):
    for trip in trips:
        vol = 0
        for passEdge in trips[trip]["pass"]:
            vol += edgesVolume[passEdge]
        trips[trip]["TotalVol"] = max(edgesVolume[e] for e in trips[trip]["pass"])
    return trips

def fixtheRoadData(road_info=None):
    net_path = "./data/ntut_network_split.net copy.xml"
    mapData = getMapData()
    edgesVolume = getEdgesVolume(road_info)
    net = SLB.net.readNet(net_path)
    for edgeId in mapData:
        if edgeId not in edgesVolume:
            edgesVolume[edgeId] = 0
    for edgeId in edgesVolume:
        if edgesVolume[edgeId] == 0:
            edgesVolume[edgeId] += round(generateEmptyEdgesVolume(net,edgesVolume,edgeId,set()))
    trips = findTrip(net,edgesVolume)
    tripsAndVol = completeTheVol(trips,edgesVolume)
    tripsAndVol = {tripId : info for tripId,info in tripsAndVol.items() if info["TotalVol"] > 0}
    return tripsAndVol
if __name__ == "__main__":
    trips_and_vol = fixtheRoadData()
    print(trips_and_vol)
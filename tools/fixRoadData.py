import os
import xml.etree.ElementTree as ET
from math import floor
import sumolib as SLB
import selectRoad as ST

# ─── 可調參數（env override） ──────────────────────────────────────────────
# 每多經過一條 edge，原始 VD 觀測流量沿路衰減的係數（0..1）。
# 物理意義：車輛在中途路口分流，遠端 edge 看到的流量會比起點低。
# 0.75 是現場手調出來的暫定值，沒有實測校準；若有實際 OD 對照表，應該以
# `route_vol[k] / origin_vol` 的中位數重新估計。
VOL_DECAY_PER_EDGE = float(os.environ.get("TRAFFICVISION_VOL_DECAY", "0.75"))

# 缺資料 edge 的車流由上游 edge 遞迴反推時，每跳衰減的權重（0..1）。
# 越小 → 缺資料 edge 越接近 0；越大 → 缺資料 edge 越接近上游平均。
# 0.35 是原 prototype 設定，無依據；可考慮改成跨資料集的 cross-validation 最佳值。
UPSTREAM_BACKFILL_ALPHA = float(os.environ.get("TRAFFICVISION_BACKFILL_ALPHA", "0.35"))

# DFS 構造 trip 時，單一 trip 內最多串接幾條 edge。
# 太短：把原本的長路徑切成多個 trip → 同一輛車重複進系統。
# 太長：DFS 在迴圈或網狀路口下會回溯非常久。
# 北科大周邊區網約 150 個 edge，10 是憑經驗的折衷值。
TRIP_MAX_EDGES = int(os.environ.get("TRAFFICVISION_TRIP_MAX_EDGES", "10"))


def getMapData():
    path = "./data/ntut_network_split.net.xml"
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
                    decayed_vol = int(float(roadInfo[name]["TotalVol"])) * pow(VOL_DECAY_PER_EDGE, passNum)
                    if edgeId not in finalInfo:
                        finalInfo[edgeId] = [decayed_vol, 1]
                    else:
                        finalInfo[edgeId][0] += decayed_vol
                        finalInfo[edgeId][1] += 1
                    passNum += 1
    for edgeId in finalInfo:
        finalInfo[edgeId] = finalInfo[edgeId][0] // finalInfo[edgeId][1]
    return finalInfo
#利用遞迴補足缺失edge的車流資訊
def generateEmptyEdgesVolume(net,edgesVolume,roadId,visited):
    ALPHA = UPSTREAM_BACKFILL_ALPHA
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
    LIMITPASSEDGE = TRIP_MAX_EDGES
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
        trips[trip]["TotalVol"] = vol
    return trips

def fixtheRoadData():
    net_path = "./data/ntut_network_split.net.xml"
    mapData = getMapData()
    edgesVolume = getEdgesVolume()
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
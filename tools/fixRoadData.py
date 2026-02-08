import xml.etree.ElementTree as ET
from math import floor
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
#利用遞迴補足缺失edge的車流資訊
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
#深層搜尋
def deepSearch(id,net,trips,edgesVolume,Edges,remainingEdges,mode:int):
    LIMITPASSEDGE = 4
    for edge in Edges:
        edgeId = edge.getID()
        if edgeId in remainingEdges:
            if mode == 0:
                Edges = net.getEdge(edgeId).getFromNode().getIncoming()
                trips["trip:" + id]["pass"].insert(0,edgeId)
            elif mode == 1:
                Edges = net.getEdge(edgeId).getToNode().getOutgoing()
                trips["trip:" + id]["pass"].append(edgeId)
            remainingEdges.remove(edgeId)
            trips = deepSearch(id,net,trips,edgesVolume,Edges,remainingEdges,mode)
            break
    return trips
#生成trips
def findTrip(net,edgesVolume):
    trips = {}
    tripId = 1
    remainingEdges = set(edgesVolume.keys())
    while len(remainingEdges) > 0:
        id = str(tripId)
        trips["trip:" + id] = {"pass" : [] , "volume" : 0}
        startEdge = next(iter(remainingEdges))
        trips["trip:" + id]["pass"].append(startEdge)
        remainingEdges.remove(startEdge)
        inComingEdges = net.getEdge(startEdge).getFromNode().getIncoming()
        outComingEdges = net.getEdge(startEdge).getToNode().getOutgoing()
        trips = deepSearch(id,net,trips,edgesVolume,inComingEdges,remainingEdges,0)
        trips = deepSearch(id,net,trips,edgesVolume,outComingEdges,remainingEdges,1)
        tripId += 1
    return trips
#完成每個trip的車流
def completeTheVol(trips,edgesVolume):
    for trip in trips:
        vol = 0
        for passEdge in trips[trip]["pass"]:
            vol += edgesVolume[passEdge]
        trips[trip]["volume"] = vol
    return trips
#加入到最終的rou檔案
import xml.etree.ElementTree as ET

def addFlow(trips_and_vol,
            output_rou_path="./data/generated.rou.xml",
            T=300):
    """
    依照 trips_and_vol 直接生成一份新的 rou 檔（包含 vType 定義）。
    """

    # ===== 1. 建立根節點 =====
    root = ET.Element("routes")

    # ===== 2. 定義車輛型別（Car2）=====
    ET.SubElement(root, "vType", {
        "id": "Car2",
        "vClass": "passenger",
        "accel": "2.6",
        "decel": "4.5",
        "length": "5.0",
        "minGap": "2.5",
        "maxSpeed": "13.9",   # 約 50 km/h
        "sigma": "0.5"
    })

    # ===== 3. 計算總車數 =====
    total_veh = 0
    for info in trips_and_vol.values():
        vol = int(info.get("volume", 0))
        if vol > 0 and info.get("pass"):
            total_veh += vol

    # 若沒有車，仍然輸出一個合法檔案
    if total_veh == 0:
        ET.ElementTree(root).write(output_rou_path, encoding="utf-8", xml_declaration=True)
        return

    gap = T / total_veh
    depart = 0.0
    veh_index = 0

    # ===== 4. 穩定排序 trip（避免每次輸出順序亂掉）=====
    def trip_key(tid):
        try:
            return int(tid.split(":")[1])
        except:
            return 10**9

    # ===== 5. 產生 vehicle =====
    for trip_id in sorted(trips_and_vol.keys(), key=trip_key):
        info = trips_and_vol[trip_id]
        path_edges = info["pass"]
        vol = int(info["volume"])

        if vol <= 0 or not path_edges:
            continue

        route_str = " ".join(path_edges)

        for _ in range(vol):
            v = ET.SubElement(root, "vehicle", {
                "id": f"{trip_id}_veh_{veh_index}",
                "type": "Car2",
                "depart": f"{depart:.2f}"
            })
            ET.SubElement(v, "route", {"edges": route_str})
            veh_index += 1
            depart += gap

    # ===== 6. 寫出新檔案 =====
    ET.ElementTree(root).write(output_rou_path, encoding="utf-8", xml_declaration=True)

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
    trips = findTrip(net,edgesVolume)
    tripsAndVol = completeTheVol(trips,edgesVolume)
    tripsAndVol = {tripId : info for tripId,info in tripsAndVol.items() if info["volume"] > 0}
    return tripsAndVol
if __name__ == "__main__":
    trips_and_vol = fixtheRoadData()
    addFlow(trips_and_vol,
            output_rou_path="./data/new_from_trips.rou.xml",
            T=300)
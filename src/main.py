import convertToRou as CTR
import selectRoad as ST
import fixRoadData as FRD
import os

# 1. 抓 VD API,只呼叫一次後重用 (省一次 API call,且避免兩端資料漂移)
roadInfo = ST.select()

# 2. 先用原始 VD 資料跑第一輪 trip + duarouter
CTR.generate_trip(roadInfo)
CTR.run_duarouter("./data/ntut_network_split.net.xml", "./data/trips.xml", "./data/output.rou.alt.xml","./data/output.rou.xml")

# 3. fixRoadData 補滿缺失 edges 並重算流量,再跑第二輪
edgesVolume = FRD.fixtheRoadData(roadInfo)
CTR.generate_trip(edgesVolume)
CTR.run_duarouter("./data/ntut_network_split.net.xml", "./data/trips.xml", "./data/final_output.rou.alt.xml","./data/final_output.rou.xml")
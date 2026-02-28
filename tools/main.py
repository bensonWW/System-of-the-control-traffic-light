import convertToRou as CTR
import selectRoad as ST
import fixRoadData as FRD
import os
CTR.generate_trip(ST.select())
CTR.run_duarouter("./data/ntut_network_split.net.xml", "./data/trips.xml", "./data/output.rou.alt.xml","./data/output.rou.xml")
edgesVolume = FRD.fixtheRoadData()
CTR.generate_trip(edgesVolume)
CTR.run_duarouter("./data/ntut_network_split.net.xml", "./data/trips.xml", "./data/final_output.rou.alt.xml","./data/final_output.rou.xml")
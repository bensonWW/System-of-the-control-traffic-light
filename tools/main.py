import convertToRou as CTR
import selectRoad as ST
import fixRoadData as FRD
CTR.generate_trip(ST.select())
CTR.run_duarouter("./data/ntut-the way.net.xml", "./data/trips.xml", "./data/output.rou.alt.xml","./data/output.rou.xml")
fixed_volumes = FRD.fixtheRoadData()
edgesVolume = FRD.fixtheRoadData()
FRD.addFlow(edgesVolume)
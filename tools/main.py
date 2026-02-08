import convertToRou as CTR
import selectRoad as ST
import fixRoadData as FRD
CTR.generate_trip(ST.select())
CTR.run_duarouter("./data/ntut-the way.net.xml", "./data/trips.xml", "./data/output.rou.alt.xml","./data/output.rou.xml")
edgesVolume = FRD.fixtheRoadData()
FRD.addFlow(edgesVolume,
            output_rou_path="./data/new_from_trips.rou.xml",
            T=300)
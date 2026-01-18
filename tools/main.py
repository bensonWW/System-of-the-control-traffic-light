import convertToRou as ctr
import selectRoad as ST
ctr.generate_trip(ST.select())
ctr.run_duarouter("./data/ntut-the way.net.xml", "./data/trips.xml", "./data/output.rou.alt.xml","./data/output.rou.xml")

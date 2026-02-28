import sys, os
sys.path.append('./tools')
import selectRoad as ST
import convertToRou as CTR
import fixRoadData as FRD

# Round 1
print("=== Round 1: selectRoad data ===")
road_info = ST.select()
print(f"Roads: {len(road_info)}")
CTR.generate_trip(road_info)
CTR.run_duarouter('./data/ntut_network_split.net copy.xml', './data/trips.xml', './data/output.rou.alt.xml', './data/output.rou.xml')
lines1 = len(open('./data/output.rou.xml').readlines())
print(f"output.rou.xml: {lines1} lines")

# Round 2
print("\n=== Round 2: fixRoadData ===")
edges_vol = FRD.fixtheRoadData()
print(f"Trips from fixRoadData: {len(edges_vol)}")
for k in list(edges_vol.keys())[:3]:
    print(f"  {k}: TotalVol={edges_vol[k]['TotalVol']}, from={edges_vol[k]['from']}, to={edges_vol[k]['to']}")

CTR.generate_trip(edges_vol)
# Check trips.xml
import xml.etree.ElementTree as ET
tree = ET.parse('./data/trips.xml')
flows = [c for c in tree.getroot() if c.tag == 'flow']
print(f"trips.xml flows: {len(flows)}")
for f in flows[:3]:
    print(f"  {f.attrib['id']}: number={f.attrib.get('number','?')}")

CTR.run_duarouter('./data/ntut_network_split.net copy.xml', './data/trips.xml', './data/final_output.rou.alt.xml', './data/final_output.rou.xml')
lines2 = len(open('./data/final_output.rou.xml').readlines())
print(f"final_output.rou.xml: {lines2} lines")

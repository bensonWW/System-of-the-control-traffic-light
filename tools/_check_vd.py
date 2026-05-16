import json, glob, sys

files = sorted(glob.glob(
    r"D:\System-of-the-control-traffic-light-master\TrafficVision Design System\data\trafficData\*.json"
))
if not files:
    print("No traffic data files found")
    sys.exit(1)

with open(files[-1], encoding="utf-8") as f:
    data = json.load(f)

roads = data.get("data", {})
print(f"File: {files[-1]}")
print(f"Total roads: {len(roads)}")
print()
for name, v in sorted(roads.items(), key=lambda x: -x[1].get("MOELevel", 0)):
    print(f"  MOE={v['MOELevel']}  spd={v['AvgSpd']:6.1f}  occ={v['AvgOcc']:5.2f}  | {name}")

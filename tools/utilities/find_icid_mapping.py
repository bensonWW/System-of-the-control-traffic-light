"""
Find ICID for missing junctions by matching coordinates and road names.
"""
import csv
import math

# Unmapped junctions with their coordinates from ntut_mapping.csv
UNMAPPED = [
    {"junction_id": "cluster3086736522_655375108_655375247_655375401_#1more", "x": 177.65, "y": 556.50},
    {"junction_id": "cluster619136880_619136881_cluster_4510442270_5296993511_5296993513_619136931_#1more", "x": 838.08, "y": 68.77},
    {"junction_id": "cluster619136899_619136900_cluster_619136929_619136942", "x": 794.09, "y": 74.74},
    {"junction_id": "cluster655375228_655375240", "x": 92.90, "y": 294.21},
    {"junction_id": "cluster655375336_end_0_655375336_end_1_655375336_end_10_655375336_end_11_#8more", "x": 395.23, "y": 489.29},
    {"junction_id": "cluster_2664674491_655375287_655375288", "x": 219.17, "y": 625.81},
]

# Load timing plan data
timing_plan = []
with open('data/source/臺北市政府交通局路口時制號誌資料(20250609更新).csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timing_plan.append(row)

print(f"Loaded {len(timing_plan)} timing plan entries")

# Keywords to search in road names for NTUT area
keywords = ["忠孝", "建國", "市民", "金山", "八德", "松江", "新生", "渭水", "臨沂"]

# Filter by keywords
candidates = []
for entry in timing_plan:
    road_name = entry.get('路口名稱', '')
    for kw in keywords:
        if kw in road_name:
            candidates.append(entry)
            break

print(f"\nFound {len(candidates)} candidates with keywords")
print("\n=== Candidate Intersections ===")
for c in candidates:
    print(f"  {c['icid']}: {c['路口名稱']}")

# Specific mappings based on road name analysis
print("\n=== Suggested Mappings ===")
suggestions = [
    ("cluster655375228_655375240", "IJTJ9", "八德路一段-金山北路 (同 joinedS_655375228_655375239_655375240)"),
    ("cluster3086736522_655375108_655375247_655375401_#1more", "IKGJI", "市民二段-金山北路"),
    ("cluster619136880_...", "IJHKR", "忠孝東三段-建國南一段"),
    ("cluster619136899_...619136942", "IJHKR", "忠孝東三段-建國南一段"),
    ("cluster655375336_end_...", "IKJJP", "市民三段-松江路"),
    ("cluster_2664674491_655375287_655375288", "IKEJW", "渭水街-松江路"),
]

for jid, icid, desc in suggestions:
    print(f"  {jid[:40]}... -> {icid} ({desc})")

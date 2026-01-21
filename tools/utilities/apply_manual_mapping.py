"""
Update sumo_json_mapping_fixed.csv with manual mappings for unmapped junctions.
"""
import csv
import shutil

# Manual mappings identified
MANUAL_MAPPINGS = [
    # (sumo_id or junction_id, icid, name)
    ("cluster655375228_655375240", "IJTJ9", "八德路一段-金山北路"),
    ("cluster3086736522_655375108_655375247_655375401_#1more", "IKGJI", "市民二段-金山北路"),
    ("joinedS_3086736522_655375108_655375247_655375401_#1more", "IKGJI", "市民二段-金山北路"),
    ("cluster619136880_619136881_cluster_4510442270_5296993511_5296993513_619136931_#1more", "IJHKR", "忠孝東三段-建國南一段"),
    ("J0", "IJHKR", "忠孝東三段-建國南一段"),
    ("cluster619136899_619136900_cluster_619136929_619136942", "IJHKR", "忠孝東三段-建國南一段"),
    ("cluster655375336_end_0_655375336_end_1_655375336_end_10_655375336_end_11_#8more", "IKJJP", "市民三段-松江路"),
    ("cluster_2664674491_655375287_655375288", "IKEJW", "渭水街-松江路"),
    ("joinedS_655375286_cluster_2664674491_655375287_655375288", "IKEJW", "渭水街-松江路"),
]

MAPPING_FILE = 'data/sumo_json_mapping_fixed.csv'

# Read existing data
rows = []
seen_ids = set()
with open(MAPPING_FILE, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)
        seen_ids.add(row['sumo_id'])

# Add/Update mappings
updates_made = 0
for sumo_id, icid, name in MANUAL_MAPPINGS:
    # Check if exists and update, or add new
    found = False
    for row in rows:
        if row['sumo_id'] == sumo_id:
            row['icid'] = icid
            row['name'] = name
            found = True
            break
    
    if not found:
        # Add new row
        rows.append({
            'junction_id': sumo_id if 'cluster' in sumo_id else '',
            'sumo_id': sumo_id,
            'icid': icid,
            'dist': '0.0',
            'name': name
        })
    updates_made += 1

# Write back
with open(MAPPING_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {updates_made} mappings in {MAPPING_FILE}")

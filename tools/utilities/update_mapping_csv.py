"""
Update sumo_json_mapping_fixed.csv with junction_id column.
Uses the junction_tls_mapping.csv to add junction IDs.
"""
import csv

# Read junction-TLS mapping
junction_to_tls = {}
tls_to_junction = {}

with open("data/legacy/junction_tls_mapping.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        jid = row['junction_id']
        tls = row['tls_id']
        junction_to_tls[jid] = tls
        if tls not in tls_to_junction:
            tls_to_junction[tls] = []
        tls_to_junction[tls].append(jid)

print(f"Loaded {len(junction_to_tls)} junction mappings")

# Read existing mapping CSV
existing_rows = []
with open("data/sumo_json_mapping_fixed.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        existing_rows.append(row)

print(f"Loaded {len(existing_rows)} existing rows")

# Add junction_id column
new_rows = []
for row in existing_rows:
    sumo_id = row['sumo_id']
    
    # Try to find junction_id
    junction_id = ""
    
    # If sumo_id is a TLS ID, find the junction(s)
    if sumo_id in tls_to_junction:
        junction_id = ";".join(tls_to_junction[sumo_id])
    # If sumo_id IS a junction ID
    elif sumo_id in junction_to_tls:
        junction_id = sumo_id
    
    new_row = {'junction_id': junction_id, **row}
    new_rows.append(new_row)

# Write updated CSV
new_fieldnames = ['junction_id'] + list(fieldnames)
with open("data/sumo_json_mapping_fixed.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print(f"\nUpdated data/sumo_json_mapping_fixed.csv with {len(new_rows)} rows")
print(f"New columns: {new_fieldnames}")

# Show sample
print("\n=== Sample rows ===")
for row in new_rows[:5]:
    print(f"  {row['junction_id'][:20] if row['junction_id'] else 'N/A':20} | {row['sumo_id'][:30]:30} | {row['icid']}")

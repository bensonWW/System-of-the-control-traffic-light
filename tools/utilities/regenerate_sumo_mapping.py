"""
Regenerate sumo_json_mapping_fixed.csv from ntut_mapping.csv
This ensures the mapping reflects current TLS IDs in the network.
"""
import csv
import os

# Read ntut_mapping.csv to get current TLS IDs and their ICID/name
ntut_mapping = []
with open('data/legacy/ntut_mapping.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ntut_mapping.append(row)

print(f"Read {len(ntut_mapping)} entries from ntut_mapping.csv")

# Group by tls_id to get all junction_ids for each TLS
tls_to_junctions = {}
for row in ntut_mapping:
    tls_id = row['tls_id']
    jid = row['junction_id']
    icid = row.get('icid', '')
    name = row.get('name', '')
    
    if tls_id not in tls_to_junctions:
        tls_to_junctions[tls_id] = {
            'junction_ids': [],
            'icid': icid,
            'name': name
        }
    tls_to_junctions[tls_id]['junction_ids'].append(jid)
    # Update icid/name if this entry has it
    if icid:
        tls_to_junctions[tls_id]['icid'] = icid
    if name:
        tls_to_junctions[tls_id]['name'] = name

# Create new mapping entries
new_mapping = []
for tls_id, data in tls_to_junctions.items():
    junction_ids = ';'.join(sorted(data['junction_ids']))
    new_mapping.append({
        'junction_id': junction_ids,
        'sumo_id': tls_id,
        'icid': data['icid'],
        'dist': '0.0',
        'name': data['name']
    })

# Sort by sumo_id
new_mapping.sort(key=lambda x: x['sumo_id'])

# Write new mapping
output_file = 'data/sumo_json_mapping_fixed.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['junction_id', 'sumo_id', 'icid', 'dist', 'name'])
    writer.writeheader()
    writer.writerows(new_mapping)

print(f"Wrote {len(new_mapping)} entries to {output_file}")

# Show entries with ICID
with_icid = [m for m in new_mapping if m['icid']]
print(f"Entries with ICID: {len(with_icid)}")

for m in new_mapping:
    status = "✓" if m['icid'] else "✗"
    print(f"  {status} {m['sumo_id'][:50]}... -> {m['icid'] or 'NO ICID'}")

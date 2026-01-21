"""
Generate a clean, validated mapping CSV file.
"""
import csv

# Define the correct mappings manually based on verification
valid_mappings = [
    # ICID, SUMO_ID, Name (Approx)
    ('IJTJ8', '655375241', '忠孝東二段--八德路一段'),
    ('IJSJD', 'GS_cluster_9383263990_9383263991_9383263992_9383263993', '忠孝東二段-金山南一段'),
    ('IJQJI', 'cluster_2038168688_655375573', '忠孝東二段-臨沂街'),
    ('IK5JW', 'cluster_2528018119_655375236', '八德路一段-新生南一段'),
    ('IKEKH', 'cluster_3431431575_4777363357_655375231_6982059025', '八德路二段-渭水路'),
    ('IKXJW', 'cluster_4374470814_4374470816_656132411_656132475', '長安東二段-松江路'),
    ('IJLJW', 'cluster_5372642442_631668976_631668981_631669076', '忠孝東二段-新生南一段'),
    ('IKGKP', 'cluster_623980090_623980094', '八德路二段-建國北一段'),
    ('IKGKP', 'cluster_623980147_623980152', '八德路二段-建國北一段'),
    ('IK9KC', 'joinedS_3086736518_655375232_655375233', '八德路二段-市民三段'),
    ('IJHKR', 'joinedS_3086736519_3086736520_631668954_631668971_#3more', '忠孝東三段-建國南一段'),
    ('IKGJI', 'joinedS_3086736522_655375108_655375247_655375401_#1more', '市民二段-金山北路'),
    ('IK7KP', 'joinedS_622618015_622618144_622618197', '建國南一段-市民三段'),
    ('IK7KP', 'joinedS_622618112_622618230', '建國南一段-市民三段'),
    ('IJTJ9', 'joinedS_655375228_655375239_655375240', '八德路一段-金山北路'),
    # Add unmatched ones that were requested
    ('IK7KP', '3208478121', '建國南一段-市民三段 (Unsignaled?)'), 
    ('IK7KP', '619136999', '建國南一段-市民三段 (Unsignaled?)'), 
]

with open('data/sumo_json_mapping.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sumo_id', 'icid', 'dist', 'name'])
    for icid, sumo_id, name in valid_mappings:
        writer.writerow([sumo_id, icid, '0.0', name])

print(f"Mapped {len(valid_mappings)} valid TLS IDs.")

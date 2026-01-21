"""
Fix invalid mappings in the CSV file by matching against real network TLS IDs.
"""
import xml.etree.ElementTree as ET
import csv
import re
import difflib

def get_real_tls_ids():
    tree = ET.parse('data/legacy/ntut-the way.net.xml')
    return set(tl.get('id') for tl in tree.findall('.//tlLogic'))

def fix_csv():
    real_ids = get_real_tls_ids()
    print(f"Loaded {len(real_ids)} real TLS IDs from network.")

    new_rows = []
    fixed_count = 0
    
    with open('data/sumo_json_mapping.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            original_id = row['sumo_id'].strip()
            icid = row['icid']
            
            if original_id in real_ids:
                new_rows.append(row)
                continue
            
            # Not found - try to match
            best_match = None
            
            # 1. Substring match
            for rid in real_ids:
                if original_id in rid: # Original is part of Real (e.g. numeric ID inside joinedS)
                    best_match = rid
                    break
                if rid in original_id: # Real is part of Original
                     best_match = rid
                     break
            
            # 2. Number set match (for clusters with different ordering)
            if not best_match:
                orig_nums = set(re.findall(r'\d+', original_id))
                for rid in real_ids:
                    real_nums = set(re.findall(r'\d+', rid))
                    if orig_nums and real_nums and orig_nums.issubset(real_nums):
                        best_match = rid
                        break
            
            if best_match:
                print(f"Fixing {original_id} -> {best_match} ({icid})")
                row['sumo_id'] = best_match
                new_rows.append(row)
                fixed_count += 1
            else:
                print(f"Could not find match for {original_id} ({icid}) - Keeping as is but marking invalid")
                # new_rows.append(row) # Keep it? Or maybe better to keep to avoid data loss
                new_rows.append(row)

    # Write back
    with open('data/sumo_json_mapping_fixed.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    
    print(f"Fixed {fixed_count} mappings. Saved to data/sumo_json_mapping_fixed.csv")

if __name__ == "__main__":
    fix_csv()

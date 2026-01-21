"""
Remove ALL references to deleted junction 655375336 from network file.
More comprehensive approach - line by line removal.
"""
import re

NET_FILE = "data/legacy/ntut_network_split.net.xml"
OUTPUT_FILE = "data/legacy/ntut_network_split.net.xml"

with open(NET_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original lines: {len(lines)}")

# Remove lines containing :655375336 references as connection source/target
new_lines = []
removed = 0
for line in lines:
    # Skip connection lines that reference :655375336
    if '<connection' in line and ':655375336' in line:
        removed += 1
        continue
    new_lines.append(line)

print(f"Removed {removed} connection lines")

# Now fix incLanes/intLanes attributes that still reference :655375336
content = ''.join(new_lines)

# Remove :655375336_* references from incLanes attribute
content = re.sub(r':655375336_\w+', '', content)

# Clean up consecutive spaces and empty lane lists
content = re.sub(r'incLanes="\s*"', 'incLanes=""', content)
content = re.sub(r'intLanes="\s*"', 'intLanes=""', content)
content = re.sub(r'  +', ' ', content)

final_count = content.count(':655375336')
print(f"Final count of ':655375336': {final_count}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Saved to: {OUTPUT_FILE}")


import xml.etree.ElementTree as ET
net_file = 'data/legacy/ntut_network_split.net.xml'
tree = ET.parse(net_file)
root = tree.getroot()
source = '49073272#4'
dest_right = '353392316#0'

print('=== Checking Right Turn ===')
conns = []
for c in root.findall('connection'):
    if c.get('from') == source and c.get('to') == dest_right:
        conns.append(f'Lane {c.get("fromLane")} -> {c.get("toLane")} (dir={c.get("dir")})')

if not conns:
    print('MISSING RIGHT TURN CONNECTION!')
else:
    for c in conns: print(c)


import xml.etree.ElementTree as ET
import os
import datetime

net_file = 'data/legacy/ntut_network_split.net.xml'

print(f'=== File Check: {net_file} ===')
try:
    mtime = os.path.getmtime(net_file)
    print(f'Last Modified: {datetime.datetime.fromtimestamp(mtime)}')
except:
    print('File not found!')

tree = ET.parse(net_file)
root = tree.getroot()

print('\n=== Checking New Connection ===')
source = '49073272#4'
dest = '668037874'
conns = []
for c in root.findall('connection'):
    if c.get('from') == source and c.get('to') == dest:
        conns.append(f"Lane {c.get('fromLane')} -> {c.get('toLane')} (dir={c.get('dir')})")

if not conns:
    print('NO CONNECTIONS FOUND!')
else:
    for c in conns: print(c)

print('\n=== Checking Signal Logic (G vs g) ===')
tl_id = 'cluster619136899_619136900_cluster_619136929_619136942'
for tl in root.findall('tlLogic'):
    if tl.get('id') == tl_id:
        print(f'TL Found: {tl_id}')
        for p in tl.findall('phase'):
            state = p.get('state')
            print(f'Phase State: {state}')

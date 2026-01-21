"""
Find unmapped TLS locations in the legacy network.
"""
import xml.etree.ElementTree as ET

tree = ET.parse('data/legacy/ntut-the way.net.xml')
root = tree.getroot()

# Unmapped TLS IDs
unmapped = [
    '3208478121',
    '619136999', 
    'J0',
    'cluster_4374470814_4374470816_656132411_656132475',
    'joinedS_655375286_cluster_2664674491_655375287_655375288'
]

print('未對應 TLS 座標資訊：')
print('=' * 80)

for tls_id in unmapped:
    # Find junction with this ID
    junction = root.find(f".//junction[@id='{tls_id}']")
    if junction is None:
        # Try to find by partial match
        for j in root.findall('.//junction'):
            jid = j.get('id')
            if tls_id in jid or jid in tls_id:
                junction = j
                break
    
    if junction is not None:
        x = junction.get('x')
        y = junction.get('y')
        jtype = junction.get('type')
        print(f'TLS: {tls_id}')
        print(f'  座標: ({x}, {y})')
        print(f'  類型: {jtype}')
        
        # Find connected edges
        inc = junction.get('incLanes', '').split() 
        if inc:
            edge_names = set(l.rsplit('_', 1)[0] for l in inc[:5] if l)
            print(f'  連接道路 ID: {list(edge_names)[:3]}')
    else:
        print(f'TLS: {tls_id}')
        print(f'  (無法找到 junction)')
    print()

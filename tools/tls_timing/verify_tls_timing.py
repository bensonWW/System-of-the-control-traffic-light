"""
Verify all TLS controllers have real timing plans applied.
"""
import xml.etree.ElementTree as ET
import csv
import re

# Load network TLS
tree = ET.parse('data/legacy/ntut-the way.net.xml')
root = tree.getroot()
network_tls = {}
for tl in root.findall('.//tlLogic'):
    tls_id = tl.get('id')
    if tls_id:
        network_tls[tls_id] = 'network_default'

# Load generated TLS
gen_tree = ET.parse('data/legacy/traffic_light.add.xml')
gen_root = gen_tree.getroot()
generated_tls = set()
for tl in gen_root.findall('.//tlLogic'):
    tls_id = tl.get('id')
    if tls_id:
        generated_tls.add(tls_id)
        network_tls[tls_id] = 'time_based'

# Load mapping
mapping = {}
with open('data/sumo_json_mapping.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sumo_id = row.get('sumo_id', '').strip()
        icid = row.get('icid', '').strip()
        name = row.get('name', '').strip()
        if sumo_id and icid:
            mapping[sumo_id] = (icid, name)

print('=' * 80)
print('Legacy 地圖 TLS 時制套用狀態確認')
print('=' * 80)
print(f'網路中 TLS 總數: {len(network_tls)}')
print(f'已套用真實時制: {len(generated_tls)}')
print(f'使用預設時制: {len(network_tls) - len(generated_tls)}')
print()
print('詳細列表:')
print('-' * 80)

applied_count = 0
default_count = 0

for tls_id in sorted(network_tls.keys()):
    status = network_tls[tls_id]
    icid = '-'
    name = '-'
    
    # Find matching ICID
    if tls_id in mapping:
        icid, name = mapping[tls_id]
    else:
        for m_id, (m_icid, m_name) in mapping.items():
            if m_id in tls_id:
                icid, name = m_icid, m_name
                break
            tls_nums = set(re.findall(r'\d+', tls_id))
            map_nums = set(re.findall(r'\d+', m_id))
            if tls_nums and map_nums and map_nums & tls_nums:
                icid, name = m_icid, m_name
                break
    
    if status == 'time_based':
        status_cn = '✅ 真實時制'
        applied_count += 1
    else:
        status_cn = '⚠️ 預設'
        default_count += 1
    
    tls_short = tls_id[:50] + '...' if len(tls_id) > 50 else tls_id
    print(f'{status_cn}  {tls_short}')
    if icid != '-':
        print(f'         -> {icid}: {name}')
    print()

print('-' * 80)
print(f'總結: {applied_count}/{len(network_tls)} 路口已套用真實時制 ({applied_count/len(network_tls)*100:.1f}%)')
print('=' * 80)

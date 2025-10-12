import xml.etree.ElementTree as ET
import csv
import datetime
from collections import defaultdict

def extract_traffic_lights(xml_file_path, tree):
    """
    提取紅綠燈資訊
    """
    root = tree.getroot()
    
    traffic_lights = []
    
    # 尋找交通號誌定義
    for tlLogic in root.findall('tlLogic'):
        tl_id = tlLogic.get('id', 'N/A')
        tl_type = tlLogic.get('type', 'N/A')
        programID = tlLogic.get('programID', '0')
        offset = tlLogic.get('offset', '0')
        
        # 提取各個相位
        phases = []
        for phase in tlLogic.findall('phase'):
            phase_duration = phase.get('duration', '0')
            phase_state = phase.get('state', 'N/A')
            phase_minDur = phase.get('minDur', phase_duration)
            phase_maxDur = phase.get('maxDur', phase_duration)
            
            phases.append({
                'duration': float(phase_duration),
                'state': phase_state,
                'minDur': float(phase_minDur),
                'maxDur': float(phase_maxDur),
                'description': decode_traffic_light_state(phase_state)
            })
        
        traffic_light_info = {
            'id': tl_id,
            'type': tl_type,
            'programID': programID,
            'offset': float(offset),
            'phases': phases,
            'total_cycle_time': sum(phase['duration'] for phase in phases),
            'phase_count': len(phases)
        }
        
        traffic_lights.append(traffic_light_info)
    
    return traffic_lights

def extract_edge_names(tree):
    """
    提取所有 edge 的 name 屬性
    """
    root = tree.getroot()
    
    edge_names = {}
    
    # 尋找所有 edge 元素
    for edge in root.findall('edge'):
        edge_id = edge.get('id', 'N/A')
        edge_name = edge.get('name', '')  # 如果沒有 name 屬性則為空字串
        edge_names[edge_id] = edge_name
    
    return edge_names

def decode_traffic_light_state(state_string):
    """
    解碼紅綠燈狀態字串
    G = 綠燈, y = 黃燈, r = 紅燈, R = 紅燈（優先）
    """
    state_map = {
        'G': '綠燈',
        'g': '綠燈(次要)',
        'y': '黃燈',
        'Y': '黃燈(優先)',
        'r': '紅燈',
        'R': '紅燈(優先)',
        '-': '無控制',
        's': '停止',
        'u': '未知'
    }
    
    decoded = []
    for i, char in enumerate(state_string):
        signal_state = state_map.get(char, f'未知({char})')
        decoded.append(f"信號{i+1}:{signal_state}")
    
    return ' | '.join(decoded)

def get_current_traffic_light_state(traffic_light, current_time):
    """
    根據當前時間計算紅綠燈的狀態
    """
    if not traffic_light['phases']:
        return None
    
    # 計算在週期中的位置
    cycle_time = traffic_light['total_cycle_time']
    offset = traffic_light['offset']
    
    # 考慮偏移量
    time_in_cycle = (current_time + offset) % cycle_time
    
    # 找到當前相位
    elapsed_time = 0
    current_phase = None
    remaining_time = 0
    
    for i, phase in enumerate(traffic_light['phases']):
        if elapsed_time <= time_in_cycle < elapsed_time + phase['duration']:
            current_phase = i
            remaining_time = elapsed_time + phase['duration'] - time_in_cycle
            break
        elapsed_time += phase['duration']
    
    if current_phase is None:
        current_phase = 0
        remaining_time = traffic_light['phases'][0]['duration']
    
    return {
        'current_phase': current_phase,
        'phase_info': traffic_light['phases'][current_phase],
        'remaining_time': remaining_time,
        'time_in_cycle': time_in_cycle,
        'cycle_time': cycle_time
    }

def extract_connections(tree, exclude_internal=False):
    """
    從 SUMO 網路檔案中提取所有 connection 資訊
    
    Parameters:
    tree: 已解析的 XML tree 物件
    exclude_internal: 是否排除交叉路口內部連接（以 : 開頭的路段）
    """
    root = tree.getroot()
    
    # 先提取所有 edge 的名稱
    edge_names = extract_edge_names(tree)
    
    connections = []
    for connection in root.findall('connection'):
        from_edge = connection.get('from', 'N/A')
        to_edge = connection.get('to', 'N/A')
        from_lane = connection.get('fromLane', 'N/A')
        to_lane = connection.get('toLane', 'N/A')
        via = connection.get('via', 'N/A')
        direction = connection.get('dir', 'N/A')
        state = connection.get('state', 'N/A')
        
        # 如果設定排除內部連接，且起點是內部路段（以:開頭），則跳過
        if exclude_internal and from_edge.startswith(':'):
            continue
        
        # 取得 edge 名稱
        from_edge_name = edge_names.get(from_edge, '')
        to_edge_name = edge_names.get(to_edge, '')
        
        connection_info = {
            'from_edge': f"'{from_edge}" if from_edge.startswith('-') else from_edge,
            'to_edge': f"'{to_edge}" if to_edge.startswith('-') else to_edge,
            'from_edge_name': from_edge_name,
            'to_edge_name': to_edge_name,
            'from_lane': from_lane,
            'to_lane': to_lane,
            'via': via,
            'direction': direction,
            'state': state,
            'type': '交叉路口內部' if from_edge.startswith(':') else '道路連接',
            'description': f"從 {from_edge_name if from_edge_name else from_edge} 路的第 {from_lane} 車道連接到 {to_edge_name if to_edge_name else to_edge} 路的第 {to_lane} 車道",
            'controlled_by_tl': False,  # 預設不受紅綠燈控制
            'tl_id': None,
            'tl_link_index': None
        }
        
        connections.append(connection_info)
    
    return connections

def link_connections_to_traffic_lights(connections, traffic_lights, tree):
    """
    將連接與紅綠燈關聯
    """
    root = tree.getroot()
    
    # 建立紅綠燈索引
    tl_dict = {tl['id']: tl for tl in traffic_lights}
    
    # 尋找受紅綠燈控制的連接
    for connection_elem in root.findall('connection'):
        tl_id = connection_elem.get('tl')
        link_index = connection_elem.get('linkIndex')
        
        if tl_id and link_index is not None:
            from_edge = connection_elem.get('from', 'N/A')
            to_edge = connection_elem.get('to', 'N/A')
            from_lane = connection_elem.get('fromLane', 'N/A')
            to_lane = connection_elem.get('toLane', 'N/A')
            
            # 在連接列表中找到對應的連接並更新
            for conn in connections:
                if (conn['from_edge'].strip("'") == from_edge and 
                    conn['to_edge'].strip("'") == to_edge and
                    conn['from_lane'] == from_lane and
                    conn['to_lane'] == to_lane):
                    
                    conn['controlled_by_tl'] = True
                    conn['tl_id'] = tl_id
                    conn['tl_link_index'] = int(link_index)
                    break
    
    return connections

def save_traffic_lights_to_csv(traffic_lights, output_file):
    """
    將紅綠燈資訊儲存為 CSV 檔案
    """
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['id', 'type', 'programID', 'offset', 'total_cycle_time', 'phase_count', 'phase_details']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for tl in traffic_lights:
            # 將相位資訊轉換為字串
            phase_details = []
            for i, phase in enumerate(tl['phases']):
                phase_str = f"相位{i+1}: {phase['duration']}秒 - {phase['description']}"
                phase_details.append(phase_str)
            
            row = {
                'id': tl['id'],
                'type': tl['type'],
                'programID': tl['programID'],
                'offset': tl['offset'],
                'total_cycle_time': tl['total_cycle_time'],
                'phase_count': tl['phase_count'],
                'phase_details': ' || '.join(phase_details)
            }
            writer.writerow(row)
    
    print(f"紅綠燈資訊已儲存至 {output_file}")

def save_individual_traffic_light_to_csv(traffic_light, output_file):
    """
    將單個紅綠燈資訊儲存為 CSV 檔案
    """
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['id', 'type', 'programID', 'offset', 'total_cycle_time', 'phase_count', 'phase_details']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # 將相位資訊轉換為字串
        phase_details = []
        for i, phase in enumerate(traffic_light['phases']):
            phase_str = f"相位{i+1}: {phase['duration']}秒 - {phase['description']}"
            phase_details.append(phase_str)
        
        row = {
            'id': traffic_light['id'],
            'type': traffic_light['type'],
            'programID': traffic_light['programID'],
            'offset': traffic_light['offset'],
            'total_cycle_time': traffic_light['total_cycle_time'],
            'phase_count': traffic_light['phase_count'],
            'phase_details': ' || '.join(phase_details)
        }
        writer.writerow(row)
    
    print(f"紅綠燈 {traffic_light['id']} 資訊已儲存至 {output_file}")

def save_individual_traffic_light_phases_to_csv(traffic_light, output_file):
    """
    將單個紅綠燈的詳細相位資訊儲存為 CSV 檔案
    """
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['tl_id', 'phase_number', 'duration', 'state', 'min_duration', 'max_duration', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for i, phase in enumerate(traffic_light['phases']):
            row = {
                'tl_id': traffic_light['id'],
                'phase_number': i + 1,
                'duration': phase['duration'],
                'state': phase['state'],
                'min_duration': phase['minDur'],
                'max_duration': phase['maxDur'],
                'description': phase['description']
            }
            writer.writerow(row)
    
    print(f"紅綠燈 {traffic_light['id']} 相位詳細資訊已儲存至 {output_file}")

def generate_traffic_light_timeline(traffic_lights, duration_seconds=300):
    """
    生成紅綠燈時間軸（預設5分鐘）
    """
    timeline = []
    current_time = datetime.datetime.now()
    
    for second in range(duration_seconds):
        timestamp = current_time + datetime.timedelta(seconds=second)
        time_snapshot = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'simulation_time': second,
            'traffic_lights': []
        }
        
        for tl in traffic_lights:
            tl_state = get_current_traffic_light_state(tl, second)
            if tl_state:
                tl_info = {
                    'id': tl['id'],
                    'current_phase': tl_state['current_phase'],
                    'phase_state': tl_state['phase_info']['state'],
                    'phase_description': tl_state['phase_info']['description'],
                    'remaining_time': round(tl_state['remaining_time'], 1),
                    'time_in_cycle': round(tl_state['time_in_cycle'], 1)
                }
                time_snapshot['traffic_lights'].append(tl_info)
        
        timeline.append(time_snapshot)
    
    return timeline

def save_timeline_to_csv(timeline, output_file):
    """
    將時間軸儲存為 CSV 檔案
    """
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['timestamp', 'simulation_time', 'tl_id', 'current_phase', 'phase_state', 'phase_description', 'remaining_time', 'time_in_cycle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for snapshot in timeline:
            for tl in snapshot['traffic_lights']:
                row = {
                    'timestamp': snapshot['timestamp'],
                    'simulation_time': snapshot['simulation_time'],
                    'tl_id': tl['id'],
                    'current_phase': tl['current_phase'],
                    'phase_state': tl['phase_state'],
                    'phase_description': tl['phase_description'],
                    'remaining_time': tl['remaining_time'],
                    'time_in_cycle': tl['time_in_cycle']
                }
                writer.writerow(row)
    
    print(f"紅綠燈時間軸已儲存至 {output_file}")

def save_individual_traffic_light_timeline_to_csv(traffic_light, duration_seconds, output_file):
    """
    為單個紅綠燈生成並儲存時間軸
    """
    timeline = []
    current_time = datetime.datetime.now()
    
    for second in range(duration_seconds):
        timestamp = current_time + datetime.timedelta(seconds=second)
        tl_state = get_current_traffic_light_state(traffic_light, second)
        
        if tl_state:
            row = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'simulation_time': second,
                'tl_id': traffic_light['id'],
                'current_phase': tl_state['current_phase'],
                'phase_state': tl_state['phase_info']['state'],
                'phase_description': tl_state['phase_info']['description'],
                'remaining_time': round(tl_state['remaining_time'], 1),
                'time_in_cycle': round(tl_state['time_in_cycle'], 1),
                'cycle_time': tl_state['cycle_time']
            }
            timeline.append(row)
    
    # 儲存到 CSV
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['timestamp', 'simulation_time', 'tl_id', 'current_phase', 'phase_state', 'phase_description', 'remaining_time', 'time_in_cycle', 'cycle_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in timeline:
            writer.writerow(row)
    
    print(f"紅綠燈 {traffic_light['id']} 時間軸已儲存至 {output_file}")

def save_individual_traffic_light_connections_to_csv(traffic_light_id, connections, output_file):
    """
    儲存特定紅綠燈控制的連接資訊
    """
    # 過濾出該紅綠燈控制的連接
    tl_connections = [conn for conn in connections if conn.get('tl_id') == traffic_light_id]
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['from_edge', 'to_edge', 'from_edge_name', 'to_edge_name', 'from_lane', 'to_lane', 'direction', 'state', 'via', 'type', 'description', 'controlled_by_tl', 'tl_id', 'tl_link_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for conn in tl_connections:
            writer.writerow(conn)
    
    print(f"紅綠燈 {traffic_light_id} 控制的 {len(tl_connections)} 個連接已儲存至 {output_file}")


def filter_connections(connections, from_edge=None, to_edge=None):
    """
    根據起點或終點過濾連接
    """
    filtered = connections
    
    if from_edge:
        filtered = [conn for conn in filtered if conn['from_edge'] == from_edge]
    
    if to_edge:
        filtered = [conn for conn in filtered if conn['to_edge'] == to_edge]
    
    return filtered

def save_connections_to_csv(connections, output_file):
    """
    將連接資訊儲存為 CSV 檔案
    """
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['from_edge', 'to_edge', 'from_edge_name', 'to_edge_name', 'from_lane', 'to_lane', 'direction', 'state', 'via', 'type', 'description', 'controlled_by_tl', 'tl_id', 'tl_link_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for conn in connections:
            writer.writerow(conn)
    
    print(f"連接資訊已儲存至 {output_file}")


# 使用範例
if __name__ == "__main__":
    # 讀取您的 XML 檔案
    xml_file = "./data/ntut-the way.net.xml"
    try:
        tree = ET.parse(xml_file)
        print("=== SUMO 網路分析（包含紅綠燈）===")
        traffic_lights = extract_traffic_lights(xml_file, tree)
        print(f"找到 {len(traffic_lights)} 個紅綠燈")

        #all_connections = extract_connections(tree, exclude_internal=False)
        road_connections = extract_connections(tree, exclude_internal=True)
        #print(f"道路連接: {len(road_connections)} 個，內部連接: {len(all_connections) - len(road_connections)} 個")

        road_connections = link_connections_to_traffic_lights(road_connections, traffic_lights, tree)
        controlled_connections = [conn for conn in road_connections if conn['controlled_by_tl']]
        print(f"受紅綠燈控制的連接: {len(controlled_connections)} 個")

        timeline = generate_traffic_light_timeline(traffic_lights, 300)

        save_connections_to_csv(road_connections, "road_connections.csv")
        #save_connections_to_csv(all_connections, "all_connections.csv")
        save_traffic_lights_to_csv(traffic_lights, "traffic_lights.csv")
        save_timeline_to_csv(timeline, "traffic_light_timeline.csv")

        import os
        
        for tl in traffic_lights:
            try:
                tl_id = tl['id']
                safe_tl_id = tl_id.replace('/', '_').replace('\\', '_').replace(':', '_')[:30]
                individual_output_dir = f"紅綠燈{safe_tl_id}檔案"
                if not os.path.exists(individual_output_dir):
                    os.makedirs(individual_output_dir)
                basic_info_file = os.path.join(individual_output_dir, f"tl_{safe_tl_id}_info.csv")
                save_individual_traffic_light_to_csv(tl, basic_info_file)
                phases_file = os.path.join(individual_output_dir, f"tl_{safe_tl_id}_phases.csv")
                save_individual_traffic_light_phases_to_csv(tl, phases_file)
                timeline_file = os.path.join(individual_output_dir, f"tl_{safe_tl_id}_timeline.csv")
                save_individual_traffic_light_timeline_to_csv(tl, 300, timeline_file)
                connections_file = os.path.join(individual_output_dir, f"tl_{safe_tl_id}_connections.csv")
                save_individual_traffic_light_connections_to_csv(tl_id, road_connections, connections_file)
                print("test")
            except Exception as e:
                print(f"儲存紅綠燈 {tl_id} 的個別檔案時發生錯誤: {e}")
        
        print("=== 檔案已生成 ===")
        print("- road_connections.csv, traffic_lights.csv, traffic_light_timeline.csv")
        print("- 個別紅綠燈檔案已儲存至各自的目錄中")

    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {xml_file}")
        print("請確保 ntut-the way.net.xml 檔案在同一目錄下")
    except ET.ParseError as e:
        print(f"XML 解析錯誤: {e}")
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()
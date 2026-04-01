import sys
import os
import xml.etree.ElementTree as ET

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import sumolib

def main():
    net_path = 'data/ntut_network_split.net.xml'
    net = sumolib.net.readNet(net_path)
    
    sources = []
    sinks = []
    
    for edge in net.getEdges():
        # Exclude internal edges
        if edge.getFunction() == 'internal':
            continue
            
        is_source = len(edge.getIncoming()) == 0
        is_sink = len(edge.getOutgoing()) == 0
        
        # If it has no incoming, it's a source, but we should make sure it's valid.
        if is_source:
            sources.append(edge)
        if is_sink:
            sinks.append(edge)
            
    print(f"Found {len(sources)} source edges.")
    print(f"Found {len(sinks)} sink edges.")
    
    # 1. Append to detectors.xml
    det_tree = ET.parse('data/detectors.xml')
    root = det_tree.getroot()
    
    boundary_det_ids = []
    
    for i, edge in enumerate(sources):
        det_id = f"SRC_{edge.getID().replace('#', '_')}"
        boundary_det_ids.append(det_id)
        lane = edge.getLanes()[0] # Put detector on the first lane
        new_det = ET.Element('detectorDefinition', {'id': det_id, 'lane': lane.getID(), 'pos': '5.0'})
        root.append(new_det)

    for i, edge in enumerate(sinks):
        det_id = f"SINK_{edge.getID().replace('#', '_')}"
        boundary_det_ids.append(det_id)
        lane = edge.getLanes()[0]
        # Put detector close to start of sink edge to ensure it gets hit quickly
        new_det = ET.Element('detectorDefinition', {'id': det_id, 'lane': lane.getID(), 'pos': '5.0'})
        root.append(new_det)
        
    ET.indent(root, space="    ")
    det_tree.write('data/detectors.xml', encoding='utf-8', xml_declaration=True)
    print("Added boundary detectors to data/detectors.xml")

    # 2. Append to measurements.csv
    # We will give them a huge flow (e.g. 2000 veh/hr = quite high for a lane)
    # The current measurements.csv uses time intervals roughly around 5.0, 10.0 etc. Let's just create a block
    # from 0 to 60 minutes for each detector with high flow to flood the network.
    
    with open('data/measurements.csv', 'a') as f:
        # Instead of just one time, let's create measurements for 12 intervals (1 hour, 5 min each)
        for det_id in boundary_det_ids:
            for m in range(0, 60, 5):
                # Format: Detector;Time;qPKW;vPKW
                # q=150 in 5 mins is 1800 veh/hr (very saturated)
                # v=40 km/h
                f.write(f"{det_id};{m}.00;200;40.0\n")

    print("Added fake high volume measurements to data/measurements.csv")

if __name__ == '__main__':
    main()

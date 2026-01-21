"""
Script to remove parallel edge segments from a SUMO network.
These typically represent bus lanes or duplicated road segments.
"""
import xml.etree.ElementTree as ET
import math
import os
import shutil

NET_FILE = "data/real_world.net.xml"
OUTPUT_FILE = "data/real_world_clean.net.xml"
PARALLEL_THRESHOLD = 25  # meters - edges closer than this are considered parallel

def get_edge_center_and_length(shape_str):
    """Calculate center point and length of an edge from its shape string."""
    try:
        points = []
        for p in shape_str.split():
            coords = p.split(',')
            points.append((float(coords[0]), float(coords[1])))
        
        if len(points) < 2:
            return None, 0
        
        # Calculate centroid
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        # Calculate total length
        total_len = 0
        for i in range(len(points) - 1):
            total_len += math.sqrt((points[i+1][0]-points[i][0])**2 + (points[i+1][1]-points[i][1])**2)
        
        return (cx, cy), total_len
    except:
        return None, 0

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def main():
    print(f"Loading {NET_FILE}...")
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    
    # Collect edge information
    edges = {}  # id -> (center, length, element)
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):
            shape = edge.get('shape')
            if shape:
                center, length = get_edge_center_and_length(shape)
                if center:
                    edges[edge_id] = {'center': center, 'length': length, 'elem': edge}
    
    print(f"Found {len(edges)} non-internal edges.")
    
    # Find parallel edge pairs
    edges_to_remove = set()
    edge_list = list(edges.items())
    
    for i, (id1, data1) in enumerate(edge_list):
        for id2, data2 in edge_list[i+1:]:
            c1, c2 = data1['center'], data2['center']
            
            # Check if centers are close (parallel edges)
            if distance(c1, c2) < PARALLEL_THRESHOLD:
                # Keep the longer edge, remove the shorter
                if data1['length'] >= data2['length']:
                    edges_to_remove.add(id2)
                else:
                    edges_to_remove.add(id1)
    
    print(f"\nIdentified {len(edges_to_remove)} edges to remove.")
    
    # Remove edges
    removed = 0
    for edge in list(root.findall('.//edge')):
        if edge.get('id') in edges_to_remove:
            root.remove(edge)
            removed += 1
    
    # Also remove connections referencing removed edges
    conn_removed = 0
    for conn in list(root.findall('.//connection')):
        if conn.get('from') in edges_to_remove or conn.get('to') in edges_to_remove:
            root.remove(conn)
            conn_removed += 1
    
    print(f"Removed {removed} edges and {conn_removed} connections.")
    
    # Save
    tree.write(OUTPUT_FILE, encoding='UTF-8', xml_declaration=True)
    print(f"\nSaved cleaned network to {OUTPUT_FILE}")
    
    # Update the main net file
    shutil.copy(OUTPUT_FILE, NET_FILE)
    print(f"Copied to {NET_FILE}")

if __name__ == "__main__":
    main()

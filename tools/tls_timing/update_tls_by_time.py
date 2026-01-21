"""
Update Traffic Light Timings by Time

This script regenerates the traffic_light.add.xml file with timing plans
appropriate for the current time of day and day of week.

Usage:
    python tools/update_tls_by_time.py [--network legacy|realworld]
"""
import xml.etree.ElementTree as ET
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Set

# Add tools directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from timing_schedule import get_plan_for_time, get_timing_details, _load_timing_data

# Configuration for different networks
NETWORKS = {
    "legacy": {
        "net_xml": "data/legacy/ntut-the way.net.xml",
        "output_add_xml": "data/legacy/traffic_light.add.xml",
        "mapping_csv": "data/sumo_json_mapping.csv",
    },
    "realworld": {
        "net_xml": "data/real_world.net.xml",
        "output_add_xml": "data/traffic_light.add.xml",
        "mapping_csv": "data/sumo_json_mapping.csv",
    }
}


def load_sumo_icid_mapping(csv_path: str) -> Dict[str, str]:
    """Load SUMO ID to ICID mapping from CSV file."""
    import csv
    
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Mapping CSV not found: {csv_path}")
        return mapping
    
    # Use utf-8-sig to handle BOM (Byte Order Mark) in CSV files
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sumo_id = row.get("sumo_id", "").strip()
            icid = row.get("icid", "").strip()
            if sumo_id and icid:
                mapping[sumo_id] = icid
    
    return mapping


def get_tls_state_length(root: ET.Element, tls_id: str) -> int:
    """Get the expected state length for a TLS from its connections."""
    max_index = -1
    for conn in root.findall(".//connection"):
        if conn.get("tl") == tls_id:
            link_index = conn.get("linkIndex")
            if link_index:
                max_index = max(max_index, int(link_index))
    return max_index + 1 if max_index >= 0 else 0


def get_tls_link_directions(root: ET.Element, tls_id: str) -> Dict[str, List[int]]:
    """
    Extract link indices grouped by approach direction.
    Returns dict mapping direction (N, S, E, W) to list of link indices.
    """
    import math
    
    # Build edge angle map
    edge_angles = {}
    for edge in root.findall(".//edge"):
        edge_id = edge.get("id")
        shape = edge.get("shape", "")
        if edge_id and not edge_id.startswith(":") and shape:
            parts = shape.strip().split()
            if len(parts) >= 2:
                x0, y0 = map(float, parts[0].split(','))
                x1, y1 = map(float, parts[-1].split(','))
                angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
                edge_angles[edge_id] = angle
    
    def angle_to_dir(angle: float) -> str:
        """Convert angle to cardinal direction."""
        # Normalize to 0-360
        angle = angle % 360
        if angle < 0:
            angle += 360
        # Map to N, E, S, W
        if 45 <= angle < 135:
            return "N"
        elif 135 <= angle < 225:
            return "W"
        elif 225 <= angle < 315:
            return "S"
        else:
            return "E"
    
    directions: Dict[str, List[int]] = {"N": [], "S": [], "E": [], "W": []}
    
    for conn in root.findall(".//connection"):
        if conn.get("tl") == tls_id:
            from_edge = conn.get("from")
            link_index = conn.get("linkIndex")
            if from_edge and link_index:
                angle = edge_angles.get(from_edge, 0)
                # Incoming direction is opposite of edge angle
                incoming_dir = angle_to_dir(angle + 180)
                directions[incoming_dir].append(int(link_index))
    
    return directions


def generate_tls_phases(
    tls_id: str,
    state_len: int,
    subplans: List[Dict],
    directions: Dict[str, List[int]]
) -> str:
    """
    Generate TLS phase XML based on timing plan subplans.
    
    Args:
        tls_id: SUMO TLS ID
        state_len: Length of state string
        subplans: List of phase dicts with green, yellow, allred times
        directions: Dict mapping directions to link indices
    
    Returns:
        XML string for tlLogic element
    """
    phases = []
    
    # Create phase patterns based on number of subphases
    # Common patterns: 2-phase (NS vs EW), 4-phase (N, E, S, W)
    num_phases = len(subplans)
    
    if num_phases >= 4:
        # 4-phase: N, E, S, W separately
        phase_dirs = [["N"], ["E"], ["S"], ["W"]]
    elif num_phases >= 2:
        # 2-phase: NS vs EW
        phase_dirs = [["N", "S"], ["E", "W"]]
    else:
        # Single phase: all green
        phase_dirs = [["N", "S", "E", "W"]]
    
    for i, subplan in enumerate(subplans):
        phase_idx = i % len(phase_dirs)
        green_dirs = phase_dirs[phase_idx]
        
        green_time = subplan.get("green", 30)
        yellow_time = subplan.get("yellow", 3)
        allred_time = subplan.get("allred", 2)
        
        # Build green state
        state_g = ['r'] * state_len
        for d in green_dirs:
            for idx in directions.get(d, []):
                if 0 <= idx < state_len:
                    state_g[idx] = 'G'
        
        # Build yellow state
        state_y = ['r'] * state_len
        for d in green_dirs:
            for idx in directions.get(d, []):
                if 0 <= idx < state_len:
                    state_y[idx] = 'y'
        
        # All red state
        state_r = 'r' * state_len
        
        # Add phases
        if green_time > 0:
            phases.append(f'        <phase duration="{green_time}" state="{"".join(state_g)}"/>')
        if yellow_time > 0:
            phases.append(f'        <phase duration="{yellow_time}" state="{"".join(state_y)}"/>')
        if allred_time > 0:
            phases.append(f'        <phase duration="{allred_time}" state="{state_r}"/>')
    
    # If no phases generated, create a default all-green phase
    if not phases:
        state = 'G' * state_len
        phases.append(f'        <phase duration="30" state="{state}"/>')
    
    phases_str = "\n".join(phases)
    return f'''    <tlLogic id="{tls_id}" type="static" programID="time_based" offset="0">
        <param key="tls.ignore-internal-junction-jam" value="true"/>
{phases_str}
    </tlLogic>'''


def get_network_tls_ids(root: ET.Element) -> Set[str]:
    """Get all TLS IDs defined in the network."""
    tls_ids = set()
    for tl in root.findall(".//tlLogic"):
        tls_id = tl.get("id")
        if tls_id:
            tls_ids.add(tls_id)
    return tls_ids


def update_tls_for_network(network: str, current_time: datetime = None) -> bool:
    """
    Update traffic light timing file for the specified network.
    
    Args:
        network: "legacy" or "realworld"
        current_time: Time to use for schedule lookup (defaults to system time)
    
    Returns:
        True if successful, False otherwise
    """
    if current_time is None:
        current_time = datetime.now()
    
    config = NETWORKS.get(network)
    if not config:
        print(f"Error: Unknown network '{network}'")
        return False
    
    net_xml = config["net_xml"]
    output_xml = config["output_add_xml"]
    mapping_csv = config.get("mapping_csv", "data/sumo_json_mapping.csv")
    
    print(f"=" * 60)
    print(f"Updating TLS for {network} network")
    print(f"=" * 60)
    print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ({current_time.strftime('%A')})")
    print(f"Network: {net_xml}")
    print(f"Output: {output_xml}")
    
    # Load network
    if not os.path.exists(net_xml):
        print(f"Error: Network file not found: {net_xml}")
        return False
    
    tree = ET.parse(net_xml)
    root = tree.getroot()
    
    # Load SUMO -> ICID mapping from CSV
    sumo_icid_mapping = load_sumo_icid_mapping(mapping_csv)
    print(f"Loaded {len(sumo_icid_mapping)} SUMO->ICID mappings from {mapping_csv}")
    
    # Get all TLS IDs from network
    tls_ids = get_network_tls_ids(root)
    print(f"Found {len(tls_ids)} TLS controllers in network")
    
    programs = []
    updated_count = 0
    
    for tls_id in sorted(tls_ids):
        # Check if we have a mapping for this TLS (exact or partial match)
        icid = None
        
        # Exact match first
        if tls_id in sumo_icid_mapping:
            icid = sumo_icid_mapping[tls_id]
        else:
            # Partial match: check if any mapped sumo_id is contained in the TLS ID
            import re
            for mapped_sumo_id, mapped_icid in sumo_icid_mapping.items():
                if mapped_sumo_id in tls_id:
                    icid = mapped_icid
                    break
                # Also check if numeric parts match (for cluster/joined IDs)
                tls_nums = set(re.findall(r'\d+', tls_id))
                map_nums = set(re.findall(r'\d+', mapped_sumo_id))
                if tls_nums and map_nums and map_nums & tls_nums:
                    icid = mapped_icid
                    break
        
        if icid:
            # Get the current plan for this intersection
            plan = get_plan_for_time(icid, current_time)
            segmenttype = current_time.weekday() + 1
            details = get_timing_details(icid, plan, segmenttype)
            
            if details and details.get("subplan"):
                state_len = get_tls_state_length(root, tls_id)
                directions = get_tls_link_directions(root, tls_id)
                
                print(f"  {tls_id[:50]} -> {icid} -> Plan {plan} (cycle={details.get('cycletime')}s)")
                
                program = generate_tls_phases(
                    tls_id, state_len, details["subplan"], directions
                )
                programs.append(program)
                updated_count += 1
            else:
                print(f"  {tls_id[:50]} -> {icid} -> Plan {plan} (no timing details)")
        else:
            # No mapping - skip
            pass
    
    # Write output
    if programs:
        output_content = "<additional>\n" + "\n".join(programs) + "\n</additional>"
    else:
        # Empty additional file to use network defaults
        output_content = "<additional>\n</additional>"
    
    with open(output_xml, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print(f"\nGenerated {updated_count} time-based TLS programs")
    print(f"Output written to: {output_xml}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Update TLS timings based on current time")
    parser.add_argument(
        "--network", "-n",
        choices=["legacy", "realworld", "all"],
        default="legacy",
        help="Which network to update (default: legacy)"
    )
    parser.add_argument(
        "--time", "-t",
        type=str,
        help="Override time (format: HH:MM or YYYY-MM-DD HH:MM)"
    )
    args = parser.parse_args()
    
    # Parse optional time override
    current_time = None
    if args.time:
        try:
            if " " in args.time:
                current_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M")
            else:
                today = datetime.now().date()
                time_parts = args.time.split(":")
                current_time = datetime(
                    today.year, today.month, today.day,
                    int(time_parts[0]), int(time_parts[1])
                )
            print(f"Using override time: {current_time}")
        except ValueError as e:
            print(f"Error parsing time: {e}")
            return
    
    # Update networks
    networks = ["legacy", "realworld"] if args.network == "all" else [args.network]
    
    for network in networks:
        update_tls_for_network(network, current_time)
        print()


if __name__ == "__main__":
    main()

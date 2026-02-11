import os
import sys
import math
import argparse
import traci # type: ignore
from sumolib import checkBinary # type: ignore

def get_lane_angle(shape):
    """
    Calculate the angle of the lane end (incoming direction).
    shape is a list of (x, y) tuples.
    """
    if len(shape) < 2: return 0
    p1 = shape[-2]
    p2 = shape[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

def create_arrow_shape(x, y, angle_rad, scale=1.0, movement_type='s'):
    """
    Create a small arrow polygon shape centered at x,y rotated by angle.
    Scale determines the size (requested ~0.01 for virtual).
    movement_type: 'l' (left), 'r' (right), 's' (straight), 't' (u-turn) based on logic
    """
    # Define a simple triangle pointing right (0 radians)
    # Scaled down to be very small as requested
    
    # Base size
    s = scale
    
    if movement_type == 'l':
        # Arrow pointing ~45 deg left relative to straight
        rel_angle = math.radians(45)
    elif movement_type == 'r':
        # Arrow pointing ~45 deg right
        rel_angle = math.radians(-45)
    else: # s or others
        rel_angle = 0
        
    final_angle = angle_rad + rel_angle
    
    cos_a = math.cos(final_angle)
    sin_a = math.sin(final_angle)
    
    # Local coords for a triangle/arrow
    # Tip at (s, 0), Base at (-s, s/2) and (-s, -s/2)
    local_points = [
        (s, 0),
        (-s, s*0.6),
        (-s*0.5, 0), # indented tail
        (-s, -s*0.6)
    ]
    
    world_points = []
    for lx, ly in local_points:
        wx = x + (lx * cos_a - ly * sin_a)
        wy = y + (lx * sin_a + ly * cos_a)
        world_points.append((wx, wy))
        
    return world_points

def determine_movement(from_shape, to_shape):
    """
    Determine approximate movement (Left, Right, Straight) based on shapes.
    """
    if not from_shape or not to_shape: return 's'
    
    angle_in = get_lane_angle(from_shape)
    
    # Angle out: use start of ‘to’ lane
    if len(to_shape) < 2: return 's'
    p1 = to_shape[0]
    p2 = to_shape[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_out = math.atan2(dy, dx)
    
    diff = angle_out - angle_in
    # Normalize to -pi, pi
    while diff <= -math.pi: diff += 2*math.pi
    while diff > math.pi: diff -= 2*math.pi
    
    degree = math.degrees(diff)
    
    if degree > 45: return 'l'
    if degree < -45: return 'r'
    return 's'

def run():
    print("Initializing detected Traffic Lights...")
    tls_list = traci.trafficlight.getIDList()
    
    # Store Polygons: { polgyon_id: { 'tls': tls_id, 'index': link_index } }
    # Actually simpler: Dictionary of 'tls_id' -> list of {index, polygon_id}
    tls_map = {}
    
    # Scale for "Virtual Objects"
    # User requested w/l ~ 0.01. That is extremely small (sub-pixel).
    # Since we are using Polygon, we can make it VISIBLE size, while safety is guaranteed by object type.
    # User feedback: "Still too big". Reducing significantly.
    VISUAL_SIZE = args.size
    OFFSET_FROM_STOP = 2.0 # meters back from stop line
    
    for tls in tls_list:
        links = traci.trafficlight.getControlledLinks(tls)
        # links is a list (size = phase string length)
        # each element is a list of connections [(from, to, via), ...] meant to be active for that index
        
        signals = []
        
        # We want to group by Lane to avoid stacking too many arrows on top of each other?
        # Or just draw all of them.
        
        for idx, connections in enumerate(links):
            if not connections: continue
            
            # Take representative connection
            # We usually care about unique (fromLane, direction) pairs per index.
            # Usually one index controls one movement (e.g. Straight+Right).
            # But sometimes they are split.
            
            seen_lanes = set()
            
            for conn in connections:
                from_lane = conn[0]
                to_lane = conn[1]
                
                # Identify movement
                from_shape = traci.lane.getShape(from_lane)
                to_shape = traci.lane.getShape(to_lane)
                
                move = determine_movement(from_shape, to_shape)
                
                # Check for duplicates on same lane/move?
                # Sometimes multiple 'to' lanes for same movement.
                key = (from_lane, move)
                if key in seen_lanes: continue
                seen_lanes.add(key)
                
                # Position
                # Get end of from_lane
                if len(from_shape) < 2: continue
                end_x, end_y = from_shape[-1]
                
                lane_width = traci.lane.getWidth(from_lane)
                angle_in = get_lane_angle(from_shape)
                
                # Calculate lateral offset based on lane width and movement?
                # Center is usually fine.
                # Shift backward
                center_x = end_x - max(0, -math.cos(angle_in) * OFFSET_FROM_STOP)
                center_y = end_y - max(0, -math.sin(angle_in) * OFFSET_FROM_STOP)
                
                # Create Polygon
                poly_id = f"sig_{tls}_{idx}_{move}_{from_lane}"
                
                # Color (Start Red)
                color = (255, 0, 0, 255)
                
                # Ensure 0.01 dimensions? 
                # Polygons don't strictly have "width/length" property like vehicles/PoI.
                # They are shapes. I will create a small shape.
                shape = create_arrow_shape(center_x, center_y, angle_in, scale=VISUAL_SIZE, movement_type=move)
                
                try:
                    traci.polygon.add(
                        poly_id, 
                        shape, 
                        color, 
                        fill=True, 
                        layer=200 # Top layer
                    )
                    signals.append({'id': poly_id, 'idx': idx})
                except traci.TraCIException:
                    pass # Exists?
        
        if signals:
            tls_map[tls] = signals

    print(f"Created signals for {len(tls_map)} Junctions with size {VISUAL_SIZE}m.")
    
    # Simulation Loop
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        for tls, signal_list in tls_map.items():
            try:
                state = traci.trafficlight.getRedYellowGreenState(tls)
            except: continue
            
            for sig in signal_list:
                idx = sig['idx']
                if idx >= len(state): continue
                
                s_char = state[idx].lower()
                
                # Map state to color
                if s_char == 'r':
                    color = (255, 0, 0, 255) # Red
                elif s_char == 'y':
                    color = (255, 255, 0, 255) # Yellow
                elif s_char == 'g':
                    color = (0, 255, 0, 255) # Green
                else:
                    color = (128, 128, 128, 255) # Off/Unknown
                
                traci.polygon.setColor(sig['id'], color)
        
        step += 1

    traci.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to sumocfg")
    parser.add_argument("--gui", action="store_true", default=True, help="Run with GUI")
    parser.add_argument("-s", "--size", type=float, default=0.05, help="Size of the arrow polygon (default: 0.05)")
    args = parser.parse_args()
    
    sumoBinary = checkBinary('sumo-gui') if args.gui else checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", args.config, "--start"]
    
    traci.start(sumoCmd)
    run()

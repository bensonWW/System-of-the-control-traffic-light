"""
Apply real-world timing plans to ntut_network_split.net.xml.
1. Uses update_tls_by_time_v2 to generate an additional file.
2. Uses netconvert to merge the additional file into the network.
"""
import sys
import os
import subprocess
from datetime import datetime

# Add tools to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.tls_timing import update_tls_by_time_v2

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Configuration
    net_file = os.path.join(base_dir, 'data', 'legacy', 'ntut_network_split.net.xml')
    mapping_file = os.path.join(base_dir, 'data', 'legacy', 'ntut_mapping.csv')
    output_add_file = os.path.join(base_dir, 'data', 'legacy', 'traffic_light_realworld.add.xml')
    
    print(f"Applying real-world timings to: {net_file}")
    print(f"Using mapping: {mapping_file}")
    
    # 1. Generate Additional File
    print("\n--- Generating TLS Programs ---")
    current_time = datetime.now() # Or specific time if needed
    update_tls_by_time_v2.update_tls_for_network(net_file, output_add_file, mapping_file, current_time)
    
    if not os.path.exists(output_add_file) or os.path.getsize(output_add_file) < 100:
        print("Error: Generated additional file is empty or missing.")
        return
        
    print(f"Generated {output_add_file}")
    
    # 2. Merge into Network using netconvert
    print("\n--- Merging into Network ---")
    # netconvert -s old.net.xml -i traffic_light.add.xml -o new.net.xml
    # Note: loading .add.xml as -i (input) or -a (additional)?
    # Actually, netconvert can load .add.xml as --tllogic-files or simply inputs.
    # Let's use -s for sumonet and -i for the additional logic.
    
    # Path to fixed connections (exported by fix_uncontrolled_lanes.py)
    fixed_conns = os.path.join(base_dir, 'data', 'legacy', 'fixed_connections.con.xml')
    
    cmd = [
        "netconvert",
        "-s", net_file,
        "-i", output_add_file,
        "-o", "temp_output.net.xml",
        "--tls.discard-loaded", "false"
    ]
    
    if os.path.exists(fixed_conns):
        print(f"Using fixed connections file: {fixed_conns}")
        # Use --connection-files or -x to FORCE these connections
        cmd.extend(["-x", fixed_conns])

    # [NEW] Add Manual Connections (e.g. for cluster fixes)
    manual_conns = os.path.join(base_dir, 'data', 'legacy', 'manual_connections.con.xml')
    if os.path.exists(manual_conns):
        print(f"Using manual connections file: {manual_conns}")
        # If -x exists, we might need to be careful. Netconvert accepts multiple files comma separated?
        # Or multiple -x flags? Usually comma separated for one flag.
        # But subprocess.run with multiple keys works fine usually?
        # Let's try appending. If it fails, we merge.
        # Actually safer to append to the existing -x if present? 
        # Netconvert usually takes multiple -x arguments as overwrite? Or accumulate?
        # Documentation: --connection-files FILE1,FILE2
        
        # Let's check if -x is already in cmd
        if "-x" in cmd:
            idx = cmd.index("-x")
            cmd[idx+1] += f",{manual_conns}"
        else:
            cmd.extend(["-x", manual_conns])
            
    # [NEW] Apply Lane Patch
    lane_patch = os.path.join(base_dir, 'data', 'legacy', 'patch_lane.edg.xml')
    if os.path.exists(lane_patch):
        print(f"Using lane patch file: {lane_patch}")
        cmd.extend(["-e", lane_patch])

    
    # Actually, if we use WAUT, it's an "additional" file logic. `netconvert` might NOT preserve WAUTs into the .net.xml permanently 
    # unless we are very careful. WAUTs are usually runtime `additional-files`.
    # BUT, `tlLogic` IS part of the network.
    # So if I run netconvert, it will bake the `tlLogic` into the net.xml.
    # It might NOT bake the WAUT.
    # If the user wants the network to simply *have* the real world timing as default, 
    # I should generate programID="0" and overwrite the old logic.
    
    # Let's Modify the strategy:
    # If I bake it into .net.xml, I should probably output programID="0" logic.
    # But `update_tls_by_time_v2` outputs programID="1".
    # I can simply ask the user to load the .add.xml in sumocfg? 
    # User said: "update ... TO ... net.xml".
    # This implies modifying .net.xml.
    # So I will bake the tlLogic definitions into .net.xml.
    # The WAUT logic in .add.xml might be lost or ignored by netconvert.
    # That's fine, provided the tlLogic is there. But if programID is 1, default will still be 0.
    # So, I should probably instruct netconvert to make program 1 the default? Or just tell the script to generate programID="0".
    
    # Let's hack: I will read the generated .add.xml and replace programID="1" with programID="0" 
    # and remove WAUT, BEFORE running netconvert.
    # This ensures the new logic REPLACES the old default.
    
    with open(output_add_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace programID="1" with programID="0"
    content = content.replace('programID="1"', 'programID="0"')
    
    # Remove WAUT section
    if '<WAUT' in content:
        content = content.split('<WAUT')[0] + '</additional>'
    
    with open(output_add_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Modified additional file to use programID='0' for permanent default.")

    try:
        subprocess.run(cmd, check=True)
        print("Successfully merged timing plans into network file.")
        
        # Overwrite original file
        import shutil
        if os.path.exists("temp_output.net.xml"):
            shutil.move("temp_output.net.xml", net_file)
            print(f"Overwrote original network file: {net_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running netconvert: {e}")

if __name__ == "__main__":
    main()

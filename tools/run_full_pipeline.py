"""
Run the Complete Data Processing Pipeline.

Sequence:
1. Merge Initial Data (merge_mappings)
2. Fix Cluster Network Topology (fix_cluster_tls) - PRE-FIX
3. Apply Real-World Timings (apply_realworld_timings) -> Runs `update_tls_by_time_v2` -> Runs `netconvert`
4. Fix Cluster Network Topology (fix_cluster_tls) - POST-FIX (Repair netconvert regressions)

Usage:
    python tools/run_full_pipeline.py
"""
import subprocess
import sys
import os

def run_step(cmd, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD : {cmd}")
    print(f"{'='*60}")
    try:
        if sys.platform == "win32":
            subprocess.run(cmd, shell=True, check=True)
        else:
            subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline Failed at step: {description}")
        sys.exit(1)

def main():
    print("🚀 Starting Full Pipeline Execution...")
    
    # 1. Merge Mappings
    run_step(
        "python -m tools.utilities.merge_mappings",
        "Merge Junction-TLS Mappings and ICID"
    )
    
    # 2. Pre-Fix Cluster (Ensure Update Script sees correct connections)
    run_step(
        "python -m tools.network_analysis.fix_cluster_tls -n legacy",
        "Pre-Fix Cluster TLS & Connections (Prepare for Timing Gen)"
    )
    
    # 3. Apply Timings (Generates .add.xml and merge)
    run_step(
        "python -m tools.utilities.apply_realworld_timings",
        "Generate & Apply Real-World Timings (Runs netconvert)"
    )
    
    # 4. Post-Fix Cluster (Repair netconvert regressions)
    run_step(
        "python -m tools.network_analysis.fix_cluster_tls -n legacy",
        "Post-Fix Cluster TLS & Connections (Repair netconvert drops)"
    )
    
    print(f"\n{'='*60}")
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

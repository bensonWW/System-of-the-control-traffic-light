"""
Centralized configuration for all tools.
Provides path constants and network selection mechanism.
"""
import os
import argparse

# === Path Calculation ===
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# === Network Configurations ===
# 命名規則: <map>_<type>.<ext>
# map: ntut | realworld
# type: network | config | routes | tls | mapping | poly
NETWORKS = {
    "legacy": {
        "name": "Legacy NTUT",
        "dir": os.path.join(DATA_DIR, "legacy"),
        "net_file": "ntut_network_split.net.xml",
        "net_file_original": "ntut_network.net.xml",
        "sumocfg": "ntut_config.sumocfg",
        "tls_add": "ntut_tls.add.xml",
        "mapping": "ntut_mapping.csv",
        "routes": "ntut_routes.rou.xml",
        "output_prefix": "ntut_",
    },
    "real_world": {
        "name": "Real World",
        "dir": os.path.join(DATA_DIR, "real_world"),
        "net_file": "realworld_network.net.xml",
        "net_file_clean": "realworld_network_clean.net.xml",
        "sumocfg": "realworld_config.sumocfg",
        "tls_add": "realworld_tls.add.xml",
        "mapping": "realworld_mapping.csv",
        "routes": "realworld_routes.rou.xml",
        "poly": "realworld_poly.poly.xml",
        "output_prefix": "realworld_",
    }
}

def get_network_config(network_name: str) -> dict:
    """
    Get configuration for specified network.
    
    Args:
        network_name: Either 'legacy' or 'real_world'
        
    Returns:
        dict with all paths for the selected network
    """
    if network_name not in NETWORKS:
        raise ValueError(f"Unknown network: {network_name}. Choose: {list(NETWORKS.keys())}")
    
    config = NETWORKS[network_name].copy()
    base_dir = config["dir"]
    
    # Build full paths
    config["net_path"] = os.path.join(base_dir, config["net_file"])
    config["sumocfg_path"] = os.path.join(base_dir, config["sumocfg"])
    config["tls_add_path"] = os.path.join(base_dir, config["tls_add"])
    config["mapping_path"] = os.path.join(base_dir, config["mapping"])
    config["routes_path"] = os.path.join(base_dir, config["routes"])
    
    if "net_file_original" in config:
        config["net_path_original"] = os.path.join(base_dir, config["net_file_original"])
    
    return config

def parse_network_arg():
    """
    Parse --network argument from command line.
    
    Returns:
        str: 'legacy' or 'real_world'
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--network", "-n", 
                        choices=list(NETWORKS.keys()), 
                        default="legacy", 
                        help="Which network to use (default: legacy)")
    args, _ = parser.parse_known_args()
    return args.network

def get_all_network_names():
    """Return list of available network names."""
    return list(NETWORKS.keys())

# === Convenience functions ===
def get_legacy_config():
    """Shortcut for get_network_config('legacy')"""
    return get_network_config("legacy")

def get_real_world_config():
    """Shortcut for get_network_config('real_world')"""
    return get_network_config("real_world")

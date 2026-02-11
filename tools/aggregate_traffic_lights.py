import os
import csv
import glob
import pandas as pd

def aggregate_csv_files(root_dir, suffix, output_filename):
    """
    Search for files ending with `suffix` in `root_dir` (recursively)
    and merge them into `output_filename`.
    """
    print(f"Searching for *{suffix} in {root_dir}...")
    
    # Use glob to find files recursively (might need ** for recursive if not using walk)
    # But files are in direct subdirectories: root_dir/subdir/file.csv
    # So glob(root_dir/*/*suffix) works.
    search_pattern = os.path.join(root_dir, "*", f"*{suffix}")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found for suffix '{suffix}'")
        return

    print(f"Found {len(files)} files. Merging...")
    
    # List to hold dataframes
    dfs = []
    
    for file in files:
        try:
            # Read CSV
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dfs:
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save to output file
        output_path = os.path.join(root_dir, output_filename)
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Saved aggregated file to: {output_path} ({len(combined_df)} rows)")
    else:
        print("No valid data found to merge.")

def main():
    root_dir = "紅綠燈個別檔案"
    
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return

    # Define the 4 types of files to aggregate
    tasks = [
        ("_info.csv", "aggregated_traffic_lights_info.csv"),
        ("_phases.csv", "aggregated_traffic_lights_phases.csv"),
        ("_timeline.csv", "aggregated_traffic_lights_timeline.csv"),
        ("_connections.csv", "aggregated_traffic_lights_connections.csv")
    ]
    
    for suffix, output_filename in tasks:
        print("-" * 40)
        aggregate_csv_files(root_dir, suffix, output_filename)
        print("-" * 40)

if __name__ == "__main__":
    main()

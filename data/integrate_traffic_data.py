import pandas as pd
import json
import os

# Define file paths
DATA_DIR = r"d:\System-of-the-control-traffic-light-chester\data"
TIMING_PLAN_JSON = os.path.join(DATA_DIR, "timing_plan.json")
TIMING_PLAN_TABLE_JSON = os.path.join(DATA_DIR, "timing_plan_table.json")
LOCATION_CSV = os.path.join(DATA_DIR, "臺北市政府交通局路口時制號誌資料(20250609更新).csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "integrated_timing_plan.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "integrated_timing_plan.csv")

def load_json_to_df(filepath):
    """Loads a JSON file into a Pandas DataFrame, repairing truncated files if necessary."""
    try:
        # Try utf-8-sig to handle BOM if present
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error in {filepath}: {e}. Attempting repair...")
            # Repair truncated JSON: find last '}', assuming it's a list of objects
            last_brace_index = content.rfind('}')
            if last_brace_index != -1:
                repaired_content = content[:last_brace_index+1] + ']'
                try:
                    data = json.loads(repaired_content)
                    print(f"Successfully repaired {filepath}. Lost data after byte {last_brace_index}.")
                except Exception as repair_error:
                    print(f"Repair failed: {repair_error}")
                    return pd.DataFrame()
            else:
                return pd.DataFrame()

        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    except Exception as e:
        with open("error_log.txt", "a") as log:
            log.write(f"Error loading {filepath}: {e}\n")
        return pd.DataFrame()

def load_csv_to_df(filepath):
    """Loads a CSV file info a Pandas DataFrame with BOM handling."""
    try:
        return pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(filepath, encoding='cp950')
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def main():
    print("--- Starting Traffic Data Integration ---")

    # 1. Load Data
    print(f"Loading {TIMING_PLAN_JSON}...")
    df_plan = load_json_to_df(TIMING_PLAN_JSON)
    print(f"Loaded timing_plan.json: {len(df_plan)} records")

    print(f"Loading {TIMING_PLAN_TABLE_JSON}...")
    df_table = load_json_to_df(TIMING_PLAN_TABLE_JSON)
    print(f"Loaded timing_plan_table.json: {len(df_table)} records")

    print(f"Loading {LOCATION_CSV}...")
    df_loc = load_csv_to_df(LOCATION_CSV)
    print(f"Loaded CSV: {len(df_loc)} records")

    # 2. Pre-processing (Convert icid to string and clean)
    # 2. Pre-processing (Convert icid to string and clean)
    # Normalize column names (strip whitespace and potential BOM, lowercase)
    df_plan.columns = [str(c).strip().lstrip('\ufeff') for c in df_plan.columns]
    df_table.columns = [str(c).strip().lstrip('\ufeff') for c in df_table.columns]
    df_loc.columns = [str(c).strip().lstrip('\ufeff') for c in df_loc.columns]

    with open("debug_columns.txt", "w", encoding="utf-8") as f:
        f.write(f"df_plan columns: {df_plan.columns.tolist()}\n")
        f.write(f"df_table columns: {df_table.columns.tolist()}\n")
        f.write(f"df_loc columns: {df_loc.columns.tolist()}\n")
    print("Debug info written to debug_columns.txt")
    
    # Check for keys but don't exit, just let it fail or warn
    if 'icid' not in df_plan.columns:
        print("WARNING: 'icid' column missing in df_plan. Check debug_columns.txt for available keys.")
    if 'icid' not in df_table.columns:
        print("WARNING: 'icid' column missing in df_table. Check debug_columns.txt for available keys.")

    if 'icid' in df_plan.columns:
        df_plan['icid'] = df_plan['icid'].astype(str).str.strip()
    else:
        print("WARNING: 'icid' not found in timing_plan.json")
    
    if 'icid' in df_table.columns:
        df_table['icid'] = df_table['icid'].astype(str).str.strip()
    else:
        print("WARNING: 'icid' not found in timing_plan_table.json")

    if 'icid' in df_loc.columns:
        df_loc['icid'] = df_loc['icid'].astype(str).str.strip()

    # 3. Merge A (Plan) and B (Table)
    # Note: Depending on the relationship, one might be a subset of the other or contain different details.
    # We will use an outer join to ensure we keep all timing info.
    # Check for overlapping columns to handle suffixes automatically or manually.
    print("Merging timing plan data...")
    try:
        # If columns overlap (besides icid), specify suffixes
        df_merged_timing = pd.merge(df_plan, df_table, on='icid', how='outer', suffixes=('_plan', '_table'))
        print(f"Merged Timing Data: {len(df_merged_timing)} records")
    except Exception as e:
        print(f"Merge failed: {e}")
        return

    # 4. Merge with Coordinates (Left Join onto the timing data)
    print("Merging with location data...")
    
    # Select relevant columns from CSV to avoid clutter
    # Assuming columns: icid, 經度, 緯度, 路口名稱
    loc_cols = ['icid', '經度', '緯度', '路口名稱']
    # Filter to only existing columns
    loc_cols = [c for c in loc_cols if c in df_loc.columns]
    
    try:
        df_final = pd.merge(df_merged_timing, df_loc[loc_cols], on='icid', how='left')
    except Exception as e:
        print(f"Location merge failed: {e}")
        return

    # 5. Rename & Clean
    print("Renaming columns...")
    rename_map = {
        '經度': 'Longitude',
        '緯度': 'Latitude',
        '路口名稱': 'IntersectionName',
        'icname': 'IntersectionName_Original' # Keep original if needed, or unify
    }
    df_final.rename(columns=rename_map, inplace=True)

    # 6. Reporting
    total_records = len(df_final)
    matched_coords = df_final['Latitude'].notna().sum()
    missing_coords = total_records - matched_coords

    print("\n" + "="*30)
    print("DATA INTEGRATION REPORT")
    print("="*30)
    print(f"Total Integrated Records : {total_records}")
    print(f"Records with Coordinates : {matched_coords}")
    print(f"Records MISSING Coords   : {missing_coords}")
    print("="*30 + "\n")

    # 7. Export
    print(f"Saving to {OUTPUT_JSON}...")
    # Convert valid NaNs to None for valid JSON (null)
    # Using 'records' orientation for a list of objects
    df_final.to_json(OUTPUT_JSON, orient='records', force_ascii=False, indent=2)

    print(f"Saving to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("Done.")

if __name__ == "__main__":
    main()

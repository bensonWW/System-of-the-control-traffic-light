import selectRoad as SR
import datetime
import time
import json
import os

output_dir = "data/trafficData"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_time = datetime.datetime.now()
data = SR.select()

print(f"Starting data collection. Saving to {output_dir}")

current_time = datetime.datetime.now()
filename = f"traffic_data_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
filepath = os.path.join(output_dir, filename)
record = {
    "timestamp": current_time.isoformat(),
    "data": data
    }
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=4)
print(f"[{current_time.strftime('%H:%M:%S')}] Saved data to {filename}")

while True:
    try:
        data = SR.select()
        current_time = datetime.datetime.now()
        if (current_time - start_time).total_seconds() >= 120:
            filename = f"traffic_data_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(output_dir, filename)
            record = {
                "timestamp": current_time.isoformat(),
                "data": data
                }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=4)
            print(f"[{current_time.strftime('%H:%M:%S')}] Saved data to {filename}")
            start_time = datetime.datetime.now()
        time.sleep(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(1)

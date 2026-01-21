import requests
import xml.etree.ElementTree as ET
import gzip
import shutil
import os
from datetime import datetime
from typing import Tuple, Dict, Any

def getData() -> Dict[str, Any]:
    """Original function - returns only road info."""
    road_info, _ = getDataWithTime()
    return road_info

def getDataWithTime() -> Tuple[Dict[str, Any], datetime]:
    """
    Fetch traffic data from Taipei API and extract the timestamp.
    Returns: (road_info dict, exchange_time datetime)
    """
    url = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
    response = requests.get(url)
    with open("GetVD.xml.gz", "wb") as f:
        f.write(response.content)
    with gzip.open("GetVD.xml.gz", "rb") as f_in:
        with open("GetVD.xml", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    if os.path.exists("GetVD.xml.gz"):
        os.remove("GetVD.xml.gz")

    tree = ET.parse("GetVD.xml")
    root = tree.getroot()
    
    # Extract ExchangeTime from header (usually root[0] or root[1])
    exchange_time = None
    ns = {'vd': 'http://www.motc.gov.tw/VD'}
    for child in root:
        tag_name = child.tag.split('}')[1] if '}' in child.tag else child.tag
        if tag_name == "ExchangeTime":
            time_str = child.text  # Format: 2026/01/17T20:01:02
            exchange_time = datetime.strptime(time_str, "%Y/%m/%dT%H:%M:%S")
            break
    
    if exchange_time is None:
        # Fallback to current time if not found
        exchange_time = datetime.now()
    
    # Extract road info
    roadInfo = {}
    for child1 in root[2]:
        tempDict = {}
        name = ""
        for child2 in child1:
            tag = child2.tag.split("}")[1] if '}' in child2.tag else child2.tag
            if tag == "SectionName":
                if "高" in child2.text or "快" in child2.text:
                    name = " "
                else:
                    name = child2.text
            else:
                tempDict[tag] = child2.text
        if name != " ":
            roadInfo[name] = tempDict
    
    if os.path.exists("GetVD.xml"):
        os.remove("GetVD.xml")
    
    return roadInfo, exchange_time

if __name__ == "__main__":
    road_info, api_time = getDataWithTime()
    print(f"API Time: {api_time}")
    print(f"Day of week: {api_time.strftime('%A')} ({api_time.weekday()})")
    print(f"Roads loaded: {len(road_info)}")
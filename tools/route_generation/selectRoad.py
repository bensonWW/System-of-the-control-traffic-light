import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.api_data import grabapi as gb
from tools.network_analysis import searchnetdata as sd
def inrange(lon_min,lat_min, lon_max ,lat_max,StartWgsX,StartWgsY,EndWgsX,EndWgsY):
    start_in = (lon_min <= StartWgsX <= lon_max and 
                lat_min <= StartWgsY <= lat_max)
    end_in = (lon_min <= EndWgsX <= lon_max and 
              lat_min <= EndWgsY <= lat_max)
    return start_in or end_in
def convert(roadInfo,lon_min,lat_min, lon_max ,lat_max):
    cDeltax = float(sd.getConverposition()[2]) #轉換座標
    cDeltay = float(sd.getConverposition()[3])
    for name in roadInfo:
        lon = float(roadInfo[name]["StartWgsX"])
        lat = float(roadInfo[name]["StartWgsY"])
        roadInfo[name]["StartWgsX"] = ((lon - lon_min) / (lon_max - lon_min)) * cDeltax
        roadInfo[name]["StartWgsY"] = ((lat - lat_min) / (lat_max - lat_min)) * cDeltay
        lon = float(roadInfo[name]["EndWgsX"])
        lat = float(roadInfo[name]["EndWgsY"])
        roadInfo[name]["EndWgsX"] = ((lon - lon_min) / (lon_max - lon_min)) * cDeltax
        roadInfo[name]["EndWgsY"] = ((lat - lat_min) / (lat_max - lat_min)) * cDeltay
    return roadInfo
def select(): 
    roadInfo = gb.getData()
    temp = {}
    lon_min,lat_min, lon_max ,lat_max = sd.getmapscope()
    for roadName in roadInfo:
        StartWgsX = float(roadInfo[roadName]["StartWgsX"])
        StartWgsY = float(roadInfo[roadName]["StartWgsY"])
        EndWgsX = float(roadInfo[roadName]["EndWgsX"])
        EndWgsY = float(roadInfo[roadName]["EndWgsY"])
        if inrange(lon_min,lat_min, lon_max ,lat_max,StartWgsX,StartWgsY,EndWgsX,EndWgsY):
            temp[roadName] = roadInfo[roadName]
    roadInfo = temp
    roadInfo = convert(roadInfo,lon_min,lat_min, lon_max ,lat_max)
    for roadName in roadInfo:
        roadInfo[roadName]["from"] = sd.search(roadInfo[roadName]["StartWgsX"],roadInfo[roadName]["StartWgsY"])
        roadInfo[roadName]["to"] = sd.search(roadInfo[roadName]["EndWgsX"],roadInfo[roadName]["EndWgsY"])
    return roadInfo
if __name__ == "__main__":
    print(select())
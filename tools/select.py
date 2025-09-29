import grabapi as gb
import searchnetdata as sd
def inrange(lon_min,lat_min, lon_max ,lat_max,StartWgsX,StartWgsY,EndWgsX,EndWgsY):
    start_in = (lon_min <= StartWgsX <= lon_max and 
                lat_min <= StartWgsY <= lat_max)
    end_in = (lon_min <= EndWgsX <= lon_max and 
              lat_min <= EndWgsY <= lat_max)
    return start_in or end_in
def main(): 
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
    roadName = temp
    print(roadName)
if __name__ == "__main__":
    main()
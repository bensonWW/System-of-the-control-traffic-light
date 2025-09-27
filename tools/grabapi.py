import requests
import xml.etree.ElementTree as ET
import gzip
import shutil
import os
def getData():
    url = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
    respone = requests.get(url)
    with open("GetVD.xml.gz","wb") as f:
        f.write(respone.content)
    with gzip.open("GetVD.xml.gz","rb") as f_in:
        with open("GetVD.xml","wb") as f_out:
            shutil.copyfileobj(f_in,f_out)
    if(os.path.exists("GetVD.xml.gz")):
        os.remove("GetVD.xml.gz")

    tree = ET.parse("GetVD.xml")
    root = tree.getroot()
    roadInfo = {}
    for child1 in root[2]:
        tempDict = {}
        name = ""
        for child2 in child1:
            if(child2.tag.split("}")[1] == "SectionName"):
                name = child2.text
            else:
                tempDict[child2.tag.split("}")[1]] = child2.text
        roadInfo[name] = tempDict
    if(os.path.exists("GetVD.xml")):
        os.remove("GetVD.xml")
    return roadInfo
if __name__ == "__main__":
    print(getData()["忠孝東路  八德路-林森北路"])
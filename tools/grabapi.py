import requests
import xml.etree.ElementTree as ET
import gzip
import shutil
import os
import tempfile

def getData():
    url = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
    respone = requests.get(url)

    # 使用系統臨時目錄而非 os.getcwd()，避免競爭條件與根目錄污染
    tmp_dir = tempfile.mkdtemp(prefix="trafficvision_")
    gz_path  = os.path.join(tmp_dir, "GetVD.xml.gz")
    xml_path = os.path.join(tmp_dir, "GetVD.xml")
    try:
        with open(gz_path, "wb") as f:
            f.write(respone.content)
        with gzip.open(gz_path, "rb") as f_in:
            with open(xml_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        roadInfo = {}
        for child1 in root[2]:
            tempDict = {}
            name = ""
            for child2 in child1:
                if(child2.tag.split("}")[1] == "SectionName"):
                    if("高" in child2.text or "快" in child2.text):
                        name = " "
                    else:
                        name = child2.text
                else:
                    tempDict[child2.tag.split("}")[1]] = child2.text
            if(name != " "):
                roadInfo[name] = tempDict
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return roadInfo
if __name__ == "__main__":
    print(getData())
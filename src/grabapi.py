import requests
# P4: defusedxml on untrusted external XML (Taipei VD feed). Same API as
# xml.etree.ElementTree but blocks billion-laughs / external-entity attacks.
from defusedxml.ElementTree import parse as _safe_parse
import gzip
import shutil
import os
import tempfile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _retrying_session():
    """Session with exponential backoff (1s → 2s → 4s) for the Taipei VD API."""
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://",  adapter)
    s.mount("https://", adapter)
    return s


def getData():
    url = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
    # Without a timeout, a slow or stalled Taipei VD API hangs every caller
    # downstream (selectRoad → convertToRou → main) indefinitely.
    respone = _retrying_session().get(url, timeout=30)
    respone.raise_for_status()

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

        tree = _safe_parse(xml_path)
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
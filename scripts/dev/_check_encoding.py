import requests, gzip, io, xml.etree.ElementTree as ET

url = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
print("Fetching...")
resp = requests.get(url, timeout=30)
raw = gzip.open(io.BytesIO(resp.content)).read()
root = ET.fromstring(raw)

# Find first SectionName
found = 0
for item in root.iter():
    tag = item.tag.split("}")[-1]
    if tag == "SectionName" and item.text:
        name = item.text.strip()
        print(f"Name: {name!r}")
        print(f"Repr: {[hex(ord(c)) for c in name[:10]]}")
        found += 1
        if found >= 3:
            break

# Also test prefix matching
prefixes = ["忠孝東路", "八德路", "市民大道", "建國北路", "建國南路", "新生南路", "新生北路", "松江路"]
print(f"\nPrefix '忠孝東路' hex: {[hex(ord(c)) for c in '忠孝東路']}")

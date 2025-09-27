import xml.etree.ElementTree as ET
def search(road1,road2,road3):
    if road1 not in roadInfoNametype or road2 not in roadInfoNametype or road3 not in roadInfoNametype:
        return
    road1List = []
    fromm = ""
    to = ""
    for idd in roadInfoNametype[road1]:
        fromm = roadInfoNametype[road1][idd]["from"]
        to = roadInfoNametype[road1][idd]["to"]
        road1List.append([fromm,to])
    for idd in roadInfoNametype[road2]:
        for FT in road1List:
            if roadInfoNametype[road2][idd]["from"] in FT:
                fromm = roadInfoNametype[road2][idd]["from"]
            elif roadInfoNametype[road2][idd]["to"] in FT:
                fromm = roadInfoNametype[road2][idd]["to"]
    for idd in roadInfoNametype[road3]:
        for FT in road1List:
            if roadInfoNametype[road3][idd]["from"] in FT:
                to = roadInfoNametype[road3][idd]["from"]
            elif roadInfoNametype[road3][idd]["to"] in FT:
                to = roadInfoNametype[road3][idd]["to"]
    return[fromm,to]
path = "./data/ntut-the way.net.xml"
tree = ET.parse(path)
root = tree.getroot()
roadInfoNametype = {}
for child in root:
    if child.tag == "edge" and "function" not in child.attrib and "name" in child.attrib:
        name = child.attrib["name"]
        fromm = child.attrib["from"]
        to = child.attrib["to"]
        idd = child.attrib["id"]
        if name not in roadInfoNametype:
            roadInfoNametype[name] = {}
        if idd not in roadInfoNametype[name]:
            roadInfoNametype[name][idd] = {}
        roadInfoNametype[name][idd]["from"] = fromm
        roadInfoNametype[name][idd]["to"] = to

print(search("市民大道二段","新生北路一段62巷","八德路二段"))

#if __name__ == "__main__":
    
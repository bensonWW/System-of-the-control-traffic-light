import re

def fix():
    filename = 'data/emitters.xml'
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()

    # Replace 'departLane="X"' with 'departLane="best"'
    data = re.sub(r'departLane="\d+"', 'departLane="best"', data)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(data)
    print("Successfully replaced departLane definitions.")

if __name__ == '__main__':
    fix()

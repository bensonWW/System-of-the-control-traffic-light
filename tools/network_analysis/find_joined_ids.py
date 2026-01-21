
import re

def main():
    print("Scanning data/real_world.net.xml for joinedS IDs...")
    try:
        with open("data/real_world.net.xml", "r", encoding="utf-8") as f:
            for line in f:
                if 'id="joinedS_' in line:
                    match = re.search(r'id="(joinedS_[^"]+)"', line)
                    if match:
                        jid = match.group(1)
                        if "4510442270" in jid:
                            print(f"FOUND TARGET: {jid}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


def main():
    try:
        with open("data/real_world.net.xml", "r", encoding="utf-8") as f:
            for i in range(20):
                print(f.readline().strip())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

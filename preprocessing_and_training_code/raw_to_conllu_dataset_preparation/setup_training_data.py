with open("Processed_data/bstrainingdata.txt", encoding="utf-8") as f:
    for lineno, line in enumerate(f, start=1):
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 10:
            print(f"Line {lineno} has {len(parts)} columns: {line.strip()}")

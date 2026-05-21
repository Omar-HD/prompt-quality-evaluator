import json
from collections import Counter

with open("data/raw/prompt_quality_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total samples:", len(data))

scores = Counter(item["score"] for item in data)
print("Score distribution:", scores)

print("\nExample sample:")
print(json.dumps(data[0], indent=4, ensure_ascii=False))
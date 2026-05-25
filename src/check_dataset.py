import json
from pathlib import Path
from collections import Counter, defaultdict

DATASET_PATH = Path("data/raw/prompt_quality_dataset.json")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

score_counts = Counter(item["score"] for item in data)
task_counts = Counter(item["task"] for item in data)
rationale_counts = Counter(item["rationale"] for item in data)

seen = defaultdict(set)
for item in data:
    key = (
        item["task"].strip().lower(),
        item["reference"].strip().lower(),
        item["submission"].strip().lower()
    )
    seen[key].add(item["score"])

conflicts = sum(1 for scores in seen.values() if len(scores) > 1)

print("Total examples:", len(data))
print("Score distribution:", dict(sorted(score_counts.items())))
print("Number of tasks:", len(task_counts))
print("Top 10 tasks:", task_counts.most_common(10))
print("Conflicting duplicate inputs:", conflicts)
print("Unique rationales:", len(rationale_counts))

print("\nExample sample:")
print(json.dumps(data[0], indent=2, ensure_ascii=False))

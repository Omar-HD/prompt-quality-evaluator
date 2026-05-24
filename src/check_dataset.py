import json
from pathlib import Path
from collections import Counter, defaultdict

PATH = Path("data/raw/final_prompt_quality_dataset.json")

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

score_counts = Counter(int(item["score"]) for item in data)
input_to_scores = defaultdict(set)
rationale_counts = Counter()

for item in data:
    key = (
        item["task"].strip().lower(),
        item["reference"].strip().lower(),
        item["submission"].strip().lower()
    )
    input_to_scores[key].add(int(item["score"]))
    rationale_counts[item["rationale"].strip()] += 1

conflicts = {k: v for k, v in input_to_scores.items() if len(v) > 1}
duplicates = len(data) - len(input_to_scores)

print("Dataset check report")
print("--------------------")
print("Total examples:", len(data))
print("Score distribution:", dict(score_counts))
print("Exact duplicate inputs:", duplicates)
print("Conflicting duplicate inputs:", len(conflicts))
print("Unique rationales:", len(rationale_counts))
print("Most repeated rationales:")
for rationale, count in rationale_counts.most_common(5):
    print(f"- repeated {count} times: {rationale[:120]}")

print("\nExample samples by score:")
for score in range(5):
    print(f"\nSCORE {score}")
    shown = 0
    for item in data:
        if int(item["score"]) == score:
            print("Task:", item["task"])
            print("Submission:", item["submission"])
            print("Rationale:", item["rationale"])
            shown += 1
            if shown == 2:
                break

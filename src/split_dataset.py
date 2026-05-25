import json
import random
from pathlib import Path
from collections import defaultdict, Counter

INPUT_PATH = Path("data/processed/formatted_dataset.json")

TRAIN_PATH = Path("data/processed/train.json")
VAL_PATH = Path("data/processed/validation.json")
TEST_PATH = Path("data/processed/test.json")

random.seed(42)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

by_score = defaultdict(list)

for item in data:
    output = item["output"]
    score = int(output.split("Score:")[1].split()[0])
    by_score[score].append(item)

train_data = []
val_data = []
test_data = []

for score in sorted(by_score.keys()):
    items = by_score[score]
    random.shuffle(items)

    n = len(items)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data.extend(items[:train_end])
    val_data.extend(items[train_end:val_end])
    test_data.extend(items[val_end:])

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

for path, split_data in [
    (TRAIN_PATH, train_data),
    (VAL_PATH, val_data),
    (TEST_PATH, test_data),
]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

def score_distribution(split_data):
    counts = Counter()
    for item in split_data:
        score = int(item["output"].split("Score:")[1].split()[0])
        counts[score] += 1
    return dict(sorted(counts.items()))

print("Stratified split completed")
print(f"Train: {len(train_data)}", score_distribution(train_data))
print(f"Validation: {len(val_data)}", score_distribution(val_data))
print(f"Test: {len(test_data)}", score_distribution(test_data))

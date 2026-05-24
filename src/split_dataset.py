import json
import random
from pathlib import Path

INPUT_PATH = Path("data/processed/formatted_dataset.json")

TRAIN_PATH = Path("data/processed/train.json")
VAL_PATH = Path("data/processed/validation.json")
TEST_PATH = Path("data/processed/test.json")

random.seed(42)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)

n = len(data)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

for path, split_data in [
    (TRAIN_PATH, train_data),
    (VAL_PATH, val_data),
    (TEST_PATH, test_data),
]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=4, ensure_ascii=False)

print("Dataset split completed")
print(f"Total: {n}")
print(f"Train: {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test: {len(test_data)}")

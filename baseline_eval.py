import json
from collections import Counter

# Load datasets
with open("data/processed/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("data/processed/validation.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

with open("data/processed/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Print dataset sizes
print("=" * 50)
print("DATASET SIZES")
print("=" * 50)

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Extract scores
def extract_scores(data):
    scores = []

    for sample in data:
        output = sample["output"]

        if "Score:" in output:
            try:
                score = int(
                    output.split("Score:")[1]
                    .split("\n")[0]
                    .strip()
                )

                scores.append(score)

            except:
                pass

    return scores

# Score distributions
train_scores = extract_scores(train_data)
val_scores = extract_scores(val_data)
test_scores = extract_scores(test_data)

print("\n" + "=" * 50)
print("TRAIN SCORE DISTRIBUTION")
print("=" * 50)
print(Counter(train_scores))

print("\n" + "=" * 50)
print("VALIDATION SCORE DISTRIBUTION")
print("=" * 50)
print(Counter(val_scores))

print("\n" + "=" * 50)
print("TEST SCORE DISTRIBUTION")
print("=" * 50)
print(Counter(test_scores))

# Print sample example
print("\n" + "=" * 50)
print("SAMPLE EXAMPLE")
print("=" * 50)

print(train_data[0])
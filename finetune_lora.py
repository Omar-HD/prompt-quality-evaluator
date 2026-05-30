import json
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

INPUT_FILE = "data/raw/generated_dataset.jsonl"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

INSTRUCTION = (
    "Evaluate the quality of the submitted prompt based on the rubric. "
    "Return a score from 0 to 4 and explain the reason."
)


def normalize_text(text):
    return " ".join(str(text).lower().strip().split())


def make_hash(entry):
    text = (
        entry.get("task", "")
        + " "
        + entry.get("reference", "")
        + " "
        + entry.get("submission", "")
    )
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def convert_to_project_format(entry):
    rubric = entry["rubric"]

    rubric_text = "\n".join(
        [f"{key}. {value}" for key, value in rubric.items()]
    )

    input_text = f"""Task:
{entry['task']}

Reference Prompt:
{entry['reference']}

Submitted Prompt:
{entry['submission']}

Rubric:
{rubric_text}"""

    output_text = f"""Score: {entry['score']}

Rationale:
{entry['rationale']}"""

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": output_text
    }


def extract_score(sample):
    output = sample["output"]

    try:
        return int(output.split("Score:")[1].split("\n")[0].strip())
    except Exception:
        return None


def count_duplicates(data):
    seen = set()
    duplicates = 0

    for sample in data:
        h = hashlib.md5(
            normalize_text(sample["input"]).encode("utf-8")
        ).hexdigest()

        if h in seen:
            duplicates += 1
        else:
            seen.add(h)

    return duplicates


def overlap_count(a, b):
    a_hashes = set(
        hashlib.md5(normalize_text(x["input"]).encode("utf-8")).hexdigest()
        for x in a
    )

    b_hashes = set(
        hashlib.md5(normalize_text(x["input"]).encode("utf-8")).hexdigest()
        for x in b
    )

    return len(a_hashes.intersection(b_hashes))


print("Loading raw generated dataset...")
raw_data = load_jsonl(INPUT_FILE)

print("Raw samples:", len(raw_data))

print("Removing duplicates...")
seen = set()
unique_raw = []

for entry in raw_data:
    h = make_hash(entry)

    if h not in seen:
        seen.add(h)
        unique_raw.append(entry)

print("Unique raw samples:", len(unique_raw))
print("Removed duplicates:", len(raw_data) - len(unique_raw))

print("Converting to project format...")
formatted = [
    convert_to_project_format(entry)
    for entry in unique_raw
]

print("Grouping by score...")
by_score = defaultdict(list)

for sample in formatted:
    score = extract_score(sample)

    if score in [0, 1, 2, 3, 4]:
        by_score[score].append(sample)

for score in range(5):
    print(f"Score {score}:", len(by_score[score]))

min_count = min(len(by_score[score]) for score in range(5))

print("Minimum class count:", min_count)
print("Balancing dataset...")

random.seed(RANDOM_SEED)

balanced = []

for score in range(5):
    samples = by_score[score]
    random.shuffle(samples)
    balanced.extend(samples[:min_count])

random.shuffle(balanced)

print("Balanced dataset size:", len(balanced))

train_size = int(len(balanced) * TRAIN_RATIO)
val_size = int(len(balanced) * VAL_RATIO)

train_data = balanced[:train_size]
val_data = balanced[train_size:train_size + val_size]
test_data = balanced[train_size + val_size:]

print("\nSplit sizes:")
print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

print("\nScore distributions:")
print("Train:", Counter(extract_score(x) for x in train_data))
print("Validation:", Counter(extract_score(x) for x in val_data))
print("Test:", Counter(extract_score(x) for x in test_data))

print("\nDuplicate check:")
print("Train duplicates:", count_duplicates(train_data))
print("Validation duplicates:", count_duplicates(val_data))
print("Test duplicates:", count_duplicates(test_data))

print("\nOverlap check:")
print("Train/Validation:", overlap_count(train_data, val_data))
print("Train/Test:", overlap_count(train_data, test_data))
print("Validation/Test:", overlap_count(val_data, test_data))

save_json(formatted, OUTPUT_DIR / "formatted_dataset.json")
save_json(train_data, OUTPUT_DIR / "train.json")
save_json(val_data, OUTPUT_DIR / "validation.json")
save_json(test_data, OUTPUT_DIR / "test.json")

save_json(train_data, OUTPUT_DIR / "train_training.json")
save_json(val_data, OUTPUT_DIR / "validation_training.json")
save_json(test_data, OUTPUT_DIR / "test_training.json")

print("\nSaved files to data/processed/")
import json
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

ROOT = Path(".")
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_FORMATTED = ROOT / "data" / "processed" / "formatted_dataset.json"
ORIGINAL_FORMATTED = (
    ROOT / "data" / "raw" / "original_dataset" / "data" / "processed" / "formatted_dataset.json"
)
CHAT_BATCHES_DIR = ROOT / "data" / "raw" / "chat_batches"

INSTRUCTION = (
    "Evaluate the quality of the submitted prompt based on the rubric. "
    "Return a score from 0 to 4 and explain the reason."
)


def normalize_text(text):
    return " ".join(str(text).lower().strip().split())


def hash_text(text):
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipped invalid JSON line in:", path)

    return rows


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def convert_raw_to_project_format(entry):
    rubric = entry.get("rubric", {})

    if isinstance(rubric, dict):
        rubric_text = "\n".join([f"{k}. {v}" for k, v in rubric.items()])
    else:
        rubric_text = str(rubric)

    input_text = f"""Task:
{entry.get('task', '')}

Reference Prompt:
{entry.get('reference', '')}

Submitted Prompt:
{entry.get('submission', '')}

Rubric:
{rubric_text}"""

    output_text = f"""Score: {entry.get('score')}

Rationale:
{entry.get('rationale', '')}"""

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": output_text,
    }


def is_project_format(sample):
    return (
        isinstance(sample, dict)
        and "instruction" in sample
        and "input" in sample
        and "output" in sample
    )


def extract_score(sample):
    try:
        return int(sample["output"].split("Score:")[1].split("\n")[0].strip())
    except Exception:
        return None


def sample_hash(sample):
    return hash_text(sample["input"])


def load_project_dataset(path, dataset_name):
    if not path.exists():
        print(f"Missing {dataset_name}:", path)
        return []

    print(f"Loading {dataset_name}:", path)
    data = load_json(path)

    clean = []

    for item in data:
        if is_project_format(item):
            clean.append(item)
        else:
            clean.append(convert_raw_to_project_format(item))

    print(f"{dataset_name} samples:", len(clean))
    return clean


def load_chat_batches():
    if not CHAT_BATCHES_DIR.exists():
        print("No chat_batches directory found:", CHAT_BATCHES_DIR)
        return []

    all_rows = []

    for path in sorted(CHAT_BATCHES_DIR.glob("*.jsonl")):
        print("Loading chat batch:", path)
        rows = load_jsonl(path)
        converted = [convert_raw_to_project_format(x) for x in rows]
        print("  rows:", len(converted))
        all_rows.extend(converted)

    print("Total chat batch samples:", len(all_rows))
    return all_rows


def count_duplicates(data):
    seen = set()
    duplicates = 0

    for sample in data:
        h = sample_hash(sample)

        if h in seen:
            duplicates += 1
        else:
            seen.add(h)

    return duplicates


def overlap_count(a, b):
    a_hashes = set(sample_hash(x) for x in a)
    b_hashes = set(sample_hash(x) for x in b)
    return len(a_hashes.intersection(b_hashes))


print("=" * 60)
print("MERGING DATASETS")
print("=" * 60)

all_samples = []

api_samples = load_project_dataset(API_FORMATTED, "Claude API processed dataset")
original_samples = load_project_dataset(ORIGINAL_FORMATTED, "Original data(8) dataset")
chat_samples = load_chat_batches()

all_samples.extend(api_samples)
all_samples.extend(original_samples)
all_samples.extend(chat_samples)

print("\nTotal before cleaning:", len(all_samples))

valid = []

for sample in all_samples:
    score = extract_score(sample)

    if score in [0, 1, 2, 3, 4]:
        valid.append(sample)

print("Valid samples:", len(valid))

seen = set()
deduped = []

for sample in valid:
    h = sample_hash(sample)

    if h not in seen:
        seen.add(h)
        deduped.append(sample)

print("After dedup:", len(deduped))
print("Removed duplicates:", len(valid) - len(deduped))

by_score = defaultdict(list)

for sample in deduped:
    by_score[extract_score(sample)].append(sample)

print("\nAvailable per score:")
for score in range(5):
    print(f"Score {score}:", len(by_score[score]))

min_count = min(len(by_score[s]) for s in range(5))
print("\nMinimum class count:", min_count)

random.seed(RANDOM_SEED)

balanced = []

for score in range(5):
    samples = by_score[score]
    random.shuffle(samples)
    balanced.extend(samples[:min_count])

random.shuffle(balanced)

print("Balanced total:", len(balanced))

train, val, test = [], [], []

for score in range(5):
    score_samples = [x for x in balanced if extract_score(x) == score]
    random.shuffle(score_samples)

    n = len(score_samples)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train.extend(score_samples[:train_end])
    val.extend(score_samples[train_end:val_end])
    test.extend(score_samples[val_end:])

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)


def dist(data):
    return Counter(extract_score(x) for x in data)


print("\nFinal split sizes:")
print("Train:", len(train), dist(train))
print("Validation:", len(val), dist(val))
print("Test:", len(test), dist(test))

print("\nDuplicate check:")
print("Train duplicates:", count_duplicates(train))
print("Validation duplicates:", count_duplicates(val))
print("Test duplicates:", count_duplicates(test))

print("\nOverlap check:")
print("Train/Validation:", overlap_count(train, val))
print("Train/Test:", overlap_count(train, test))
print("Validation/Test:", overlap_count(val, test))

save_json(balanced, OUTPUT_DIR / "formatted_dataset.json")
save_json(train, OUTPUT_DIR / "train.json")
save_json(val, OUTPUT_DIR / "validation.json")
save_json(test, OUTPUT_DIR / "test.json")

save_json(train, OUTPUT_DIR / "train_training.json")
save_json(val, OUTPUT_DIR / "validation_training.json")
save_json(test, OUTPUT_DIR / "test_training.json")

print("\nSaved merged clean dataset to:")
print(OUTPUT_DIR)
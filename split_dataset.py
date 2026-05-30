import json
import hashlib

TRAIN_FILE = "data/processed/train.json"
VAL_FILE = "data/processed/validation.json"
TEST_FILE = "data/processed/test.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return " ".join(text.lower().strip().split())


def make_hash(sample):
    text = normalize_text(sample["input"])
    return hashlib.md5(text.encode("utf-8")).hexdigest()


train = load_json(TRAIN_FILE)
val = load_json(VAL_FILE)
test = load_json(TEST_FILE)

train_hashes = set(make_hash(x) for x in train)
val_hashes = set(make_hash(x) for x in val)
test_hashes = set(make_hash(x) for x in test)

train_test_overlap = train_hashes.intersection(test_hashes)
train_val_overlap = train_hashes.intersection(val_hashes)
val_test_overlap = val_hashes.intersection(test_hashes)

print("Train samples:", len(train))
print("Validation samples:", len(val))
print("Test samples:", len(test))

print("\nExact input overlap:")
print("Train/Test overlap:", len(train_test_overlap))
print("Train/Validation overlap:", len(train_val_overlap))
print("Validation/Test overlap:", len(val_test_overlap))
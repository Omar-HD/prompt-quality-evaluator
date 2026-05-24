import json
from pathlib import Path
from collections import defaultdict, Counter

GENERATED_PATH = Path("data/raw/prompt_quality_dataset.json")
MANUAL_PATH = Path("data/raw/manual_refined_samples.json")
OUTPUT_PATH = Path("data/raw/final_prompt_quality_dataset.json")

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def input_key(item):
    return (
        item["task"].strip().lower(),
        item["reference"].strip().lower(),
        item["submission"].strip().lower()
    )

def main():
    generated = load_json(GENERATED_PATH)
    manual = load_json(MANUAL_PATH)

    # Manual examples first: if duplicate exists, keep the manually refined version
    combined = manual + generated

    kept = []
    seen = {}
    conflicts_removed = 0
    duplicates_removed = 0

    for item in combined:
        key = input_key(item)
        score = int(item["score"])

        if key in seen:
            old_score = int(seen[key]["score"])
            if old_score != score:
                conflicts_removed += 1
            else:
                duplicates_removed += 1
            continue

        seen[key] = item
        kept.append(item)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=4, ensure_ascii=False)

    print("Merge + clean completed")
    print(f"Generated examples loaded: {len(generated)}")
    print(f"Manual refined examples loaded: {len(manual)}")
    print(f"Final examples saved: {len(kept)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Conflicting labels removed: {conflicts_removed}")
    print("Score distribution:", dict(Counter(int(x["score"]) for x in kept)))

if __name__ == "__main__":
    main()

import json
from pathlib import Path

SPLITS = ["train", "validation", "test"]

def make_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

for split in SPLITS:
    input_path = Path(f"data/processed/{split}.json")
    output_path = Path(f"data/processed/{split}_training.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = [{"text": make_prompt(item)} for item in data]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

    print(f"{split}: saved {len(converted)} samples to {output_path}")

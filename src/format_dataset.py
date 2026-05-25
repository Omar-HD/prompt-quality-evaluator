import json
from pathlib import Path

INPUT_PATH = Path("data/raw/prompt_quality_dataset.json")
OUTPUT_PATH = Path("data/processed/formatted_dataset.json")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []

for item in data:
    instruction = (
        "Evaluate the quality of the submitted prompt based on the rubric. "
        "Return a score from 0 to 4 and explain the reason."
    )

    input_text = f"""
Task:
{item['task']}

Reference Prompt:
{item['reference']}

Submitted Prompt:
{item['submission']}

Rubric:
1. {item['rubric']['1']}
2. {item['rubric']['2']}
3. {item['rubric']['3']}
4. {item['rubric']['4']}
"""

    output_text = f"""
Score: {item['score']}

Rationale:
{item['rationale']}
"""

    formatted_data.append({
        "instruction": instruction,
        "input": input_text.strip(),
        "output": output_text.strip()
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=4, ensure_ascii=False)

print(f"Formatted dataset saved to: {OUTPUT_PATH}")
print(f"Total formatted samples: {len(formatted_data)}")

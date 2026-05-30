import json
import re
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TEST_FILE = "data/processed/test.json"
OUTPUT_FILE = "outputs/baseline_predictions.json"


def build_prompt(sample):
    return f"""### Instruction:
Evaluate the quality of the submitted prompt based on the rubric.

You must return the answer exactly in this format:

Score: <one number from 0 to 4>
Rationale: <short explanation>

### Input:
{sample['input']}

### Response:
"""


def extract_gold_score(output):
    match = re.search(r"Score:\s*([0-4])", output)

    if match:
        return int(match.group(1))

    return None


def extract_predicted_score(text):

    patterns = [

        r"Score:\s*([0-4])",

        r"(?:score|rating)\s*(?:is|:|=)?\s*([0-4])",

        r"(?:receives|gets|deserves)\s*(?:a\s*)?(?:score\s*of\s*)?([0-4])",

        r"(?:rate|rated)\s*(?:this\s*prompt\s*)?(?:as\s*)?([0-4])",

        r"\b([0-4])\s*/\s*4\b",

        r"\b([0-4])\s*out\s*of\s*4\b",

        r"\bscore\s+of\s+([0-4])\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return None


print("Loading test data...")

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)



print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

results = []

for i, sample in enumerate(
    tqdm(test_data, desc="Running baseline evaluation")
):

    prompt = build_prompt(sample)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    generated_response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    gold_score = extract_gold_score(
        sample["output"]
    )

    predicted_score = extract_predicted_score(
        generated_response
    )

    results.append({
        "id": i,
        "gold_score": gold_score,
        "predicted_score": predicted_score,
        "input": sample["input"],
        "gold_output": sample["output"],
        "baseline_response": generated_response
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

valid_results = [

    r for r in results

    if r["gold_score"] is not None
    and r["predicted_score"] is not None
]

y_true = [
    r["gold_score"]
    for r in valid_results
]

y_pred = [
    r["predicted_score"]
    for r in valid_results
]

print("\nSaved predictions to:", OUTPUT_FILE)

print("Total test samples:", len(results))

print("Parsed predictions:", len(valid_results))

if valid_results:

    print(
        "Accuracy:",
        accuracy_score(y_true, y_pred)
    )

    print(
        "MAE:",
        mean_absolute_error(y_true, y_pred)
    )

    print(
        "QWK:",
        cohen_kappa_score(
            y_true,
            y_pred,
            weights="quadratic"
        )
    )

else:

    print("No valid predictions parsed.")
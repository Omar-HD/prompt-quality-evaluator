import json

BASELINE_FILE = "outputs/baseline_predictions.json"
FINETUNED_FILE = "outputs/finetuned_predictions.json"


with open(BASELINE_FILE, "r", encoding="utf-8") as f:
    baseline_results = json.load(f)

with open(FINETUNED_FILE, "r", encoding="utf-8") as f:
    finetuned_results = json.load(f)


print("=" * 70)
print("BASELINE VS FINE-TUNED COMPARISON")
print("=" * 70)

for i in range(5):

    base = baseline_results[i]
    fine = finetuned_results[i]

    print(f"\nExample #{i + 1}")
    print("-" * 70)

    print("Gold Score:")
    print(base["gold_score"])

    print("\nBaseline Prediction:")
    print(base["predicted_score"])

    print("\nFine-tuned Prediction:")
    print(fine["predicted_score"])

    print("\nBaseline Response:")
    print(base["baseline_response"][:400])

    print("\nFine-tuned Response:")
    print(fine["finetuned_response"][:400])

    print("\n" + "=" * 70)
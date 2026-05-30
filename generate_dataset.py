import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAIN_FILE = "data/processed/train.json"
VAL_FILE = "data/processed/validation.json"
OUTPUT_DIR = "models/prompt_quality_lora"
MAX_SEQ_LENGTH = 2048


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for sample in data:
        text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
        rows.append({"text": text})

    return Dataset.from_list(rows)


print("Loading datasets...")
train_dataset = load_dataset(TRAIN_FILE)
val_dataset = load_dataset(VAL_FILE)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model in fp16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

model.config.use_cache = False

print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    warmup_steps=20,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    seed=42,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args
)

print("Starting fine-tuning...")
trainer.train()

print("Saving LoRA adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done. Model saved to:", OUTPUT_DIR)
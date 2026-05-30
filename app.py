import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_MODEL = "models/prompt_quality_lora"


TASK_TYPES = [
    "General Prompt",
    "Product Description",
    "Email Writing",
    "Summarization",
    "Scientific Explanation",
    "Coding Task",
    "Other (Custom)"
]


def score_label(score):
    labels = {
        0: "Very poor prompt",
        1: "Weak prompt",
        2: "Fair prompt",
        3: "Good prompt",
        4: "Excellent prompt",
    }
    return labels.get(score, "Unknown")


def score_color(score):
    colors = {
        0: "#f8d7da",
        1: "#fff3cd",
        2: "#fff9db",
        3: "#d1e7dd",
        4: "#d4edda",
    }
    return colors.get(score, "#e9ecef")


def extract_score(text):
    match = re.search(r"Score:\s*([0-4])", text, re.IGNORECASE)

    if match:
        return int(match.group(1))

    return None


def build_prompt(task_type, user_prompt):
    return f"""### Instruction:
Evaluate the quality of the submitted prompt based on the rubric. Return a score from 0 to 4 and explain the reason.

### Input:
Task:
{task_type}

Reference Prompt:
A high-quality prompt should clearly state the task, include specific requirements, specify useful constraints such as format, audience, tone, or length, and be complete enough to guide a high-quality response.

Submitted Prompt:
{user_prompt}

Rubric:
1. Does the prompt clearly state the task?
2. Does it include specific requirements or details?
3. Does it specify useful constraints such as format, audience, tone, or length?
4. Is the prompt complete and useful enough to guide a high-quality response?

### Response:
"""


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL
    )

    model.eval()

    return tokenizer, model


st.set_page_config(
    page_title="Prompt Quality Evaluator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Prompt Quality Evaluator")
st.write(
    "Evaluate prompt quality using a fine-tuned Mistral LoRA model."
)

st.divider()

task_type = st.selectbox(
    "Select Task Type",
    TASK_TYPES
)

if task_type == "Other (Custom)":
    custom_task = st.text_input(
        "Enter custom task type",
        placeholder="e.g. Legal Analysis, Marketing Campaign, Research Proposal"
    )

    if custom_task.strip():
        task_type = custom_task.strip()

user_prompt = st.text_area(
    "Enter the submitted prompt",
    height=180,
    placeholder="Write your prompt here..."
)

evaluate_button = st.button("Evaluate Prompt")

if evaluate_button:

    if task_type == "Other (Custom)":
        st.warning("Please enter a custom task type before evaluation.")

    elif not user_prompt.strip():
        st.warning("Please enter a prompt to evaluate.")

    else:
        with st.spinner("Loading model and evaluating prompt..."):
            tokenizer, model = load_model()

            prompt = build_prompt(task_type, user_prompt)

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

            model_response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            predicted_score = extract_score(model_response)

        st.divider()
        st.subheader("Result")

        if predicted_score is not None:
            st.caption("Predicted Score")
            st.markdown(f"## {predicted_score}/4")

            st.progress(predicted_score / 4)

            st.markdown(
                f"""
                <div style="
                    background-color:{score_color(predicted_score)};
                    padding:16px;
                    border-radius:10px;
                    margin-top:10px;
                    margin-bottom:20px;
                    font-weight:600;
                ">
                    {score_label(predicted_score)}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.error("Could not extract a valid score from the model response.")

        st.caption("Model Response")
        st.code(model_response)
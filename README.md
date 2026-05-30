# Prompt Quality Evaluator

## Overview

Prompt Quality Evaluator is a machine learning system that evaluates the quality of user-written prompts using a rubric-based scoring framework. The system predicts a score from 0 to 4 and generates a rationale explaining the evaluation.

The project uses instruction tuning and parameter-efficient fine-tuning (LoRA) on Mistral-7B-Instruct-v0.2 to adapt a general-purpose large language model to prompt quality assessment.

---

## Features

* Prompt quality scoring (0–4)
* Automatic rationale generation
* Fine-tuned Mistral-7B-Instruct-v0.2 model
* Streamlit web application
* Rubric-based evaluation
* Dataset preprocessing and quality assurance

---

## Dataset

The final dataset was constructed by combining multiple sources:

* Original processed dataset: 1,350 samples
* Claude API generated dataset: 926 samples
* Claude chat batches: 400 samples

After cleaning, deduplication, and balancing, the final dataset contained:

* 2,620 samples
* 524 samples per score category

Dataset split:

* Training: 1,830
* Validation: 390
* Test: 400

---

## Model

Base Model:

* Mistral-7B-Instruct-v0.2

Fine-Tuning Method:

* LoRA (Low-Rank Adaptation)
* Unsloth Framework

---

## Results

| Metric   | Baseline | Fine-Tuned |
| -------- | -------: | ---------: |
| Accuracy |   56.39% |     96.00% |
| MAE      |   0.5013 |     0.0425 |
| QWK      |   0.8490 |     0.9880 |

---

## Web Application

The project includes a Streamlit-based web application that allows users to:

* Enter prompts
* Receive a quality score
* View a rubric-based rationale

---

## Demo Video

🎥 Demo Video:

PASTE_GOOGLE_DRIVE_LINK_HERE

---

## Project Structure

```text
app.py
src/
notebooks/
```

---

## Team Members

* Mayar Basheer
* Omar El Haj Daoud
* Yosef Dabeek

---

## Technologies Used

* Python
* Hugging Face Transformers
* PEFT
* LoRA
* Unsloth
* Streamlit
* Scikit-learn
* PyTorch

---

## License

This project was developed as part of a university course project.

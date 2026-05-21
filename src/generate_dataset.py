import json
import random
from pathlib import Path

OUTPUT_PATH = Path("data/raw/prompt_quality_dataset.json")

tasks = [
    "Summarize an article",
    "Translate Arabic to English",
    "Write a professional email",
    "Explain a scientific concept",
    "Generate interview questions",
    "Write a product description",
    "Create study notes",
    "Generate social media captions",
    "Explain machine learning concepts",
    "Write a customer support response"
]

references = {
    "Summarize an article": "Summarize the following article in under 100 words focusing on the key arguments and conclusion.",
    "Translate Arabic to English": "Translate the Arabic text into fluent English while preserving the original meaning and tone.",
    "Write a professional email": "Write a polite and professional email that clearly explains the purpose and requested action.",
    "Explain a scientific concept": "Explain the concept in simple terms for beginners using examples.",
    "Generate interview questions": "Generate 5 interview questions with increasing difficulty and short expected answers.",
    "Write a product description": "Write a persuasive product description highlighting features, benefits, and target users.",
    "Create study notes": "Create concise study notes with definitions, examples, and key points.",
    "Generate social media captions": "Generate engaging captions suitable for Instagram with a friendly tone.",
    "Explain machine learning concepts": "Explain the machine learning concept using simple language and a practical example.",
    "Write a customer support response": "Write a polite customer support response that apologizes, explains, and offers a solution."
}

rubric = {
    "1": "Clarity of instruction",
    "2": "Specificity of requirements",
    "3": "Presence of constraints",
    "4": "Completeness and usefulness"
}

excellent_templates = [
    "Please {action} in under {limit} words, focusing on {focus}, and use simple language.",
    "{action} for beginners using bullet points, examples, and a clear structure.",
    "Create a clear and complete response to {action}, including {focus} and useful constraints.",
    "{action} in a professional tone, with specific requirements and concise formatting."
]

good_templates = [
    "Please {action} clearly with examples.",
    "{action} and include the most important details.",
    "Can you {action} in a simple and organized way?",
    "{action} with a clear structure and short explanation."
]

acceptable_templates = [
    "{action} clearly.",
    "Please {action} with some details.",
    "Can you help me {action}?",
    "{action} and explain briefly."
]

poor_templates = [
    "Do this.",
    "Explain.",
    "Write something.",
    "Help me.",
    "Tell me about it."
]

focuses = [
    "main ideas",
    "important details",
    "key arguments",
    "real-world examples",
    "beginner understanding",
    "advantages and disadvantages",
    "professional tone",
    "step-by-step explanation"
]

limits = [50, 100, 150, 200]

def generate_submission(score, task):
    action = task.lower()
    focus = random.choice(focuses)
    limit = random.choice(limits)

    if score == 4:
        template = random.choice(excellent_templates)
        rationale = "The prompt is excellent because it is clear, specific, complete, and includes useful constraints."
    elif score == 3:
        template = random.choice(good_templates)
        rationale = "The prompt is good because it is clear and mostly useful, but it could include stronger constraints."
    elif score == 2:
        template = random.choice(acceptable_templates)
        rationale = "The prompt is acceptable because the main intent is understandable, but it lacks detail and constraints."
    elif score == 1:
        template = random.choice(poor_templates)
        rationale = "The prompt is poor because it is vague and does not provide enough information."
    else:
        template = random.choice(poor_templates)
        rationale = "The prompt is ineffective because it does not clearly specify the task or expected output."

    submission = template.format(
        action=action,
        focus=focus,
        limit=limit
    )

    return submission, rationale

def main():
    dataset = []

    for score in range(5):
        for _ in range(240):
            task = random.choice(tasks)
            submission, rationale = generate_submission(score, task)

            entry = {
                "task": task,
                "reference": references[task],
                "submission": submission,
                "rubric": rubric,
                "score": score,
                "rationale": rationale
            }

            dataset.append(entry)

    random.shuffle(dataset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Total samples: {len(dataset)}")

if __name__ == "__main__":
    main()
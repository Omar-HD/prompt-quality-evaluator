import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

OUTPUT_PATH = Path("data/raw/prompt_quality_dataset.json")
TARGET_PER_SCORE = 240

# =========================================================
# 15 TASK CATEGORIES
# Each task has:
# - reference prompt
# - task-specific rubric
# - verb used to generate submissions
# - focus/format/audience/constraints for variation
# =========================================================

TASKS = {
    "Summarize an article": {
        "reference": "Summarize the article clearly by focusing on the main idea, key arguments, and conclusion.",
        "verb": "summarize the article",
        "rubric": {
            "1": "Does the prompt clearly request a summary?",
            "2": "Does it specify what parts of the article to focus on?",
            "3": "Does it include useful constraints such as length, format, or audience?",
            "4": "Is the prompt complete enough to guide a high-quality summary?"
        },
        "focus": ["main idea", "key arguments", "supporting evidence", "conclusion", "important details"],
        "formats": ["bullet points", "one paragraph", "numbered list", "short paragraph"],
        "audiences": ["beginners", "students", "general readers", "busy readers"],
        "constraints": ["under 100 words", "under 150 words", "in 5 bullet points", "in simple language"]
    },

    "Translate Arabic to English": {
        "reference": "Translate the Arabic text into fluent English while preserving meaning, tone, and context.",
        "verb": "translate the Arabic text into English",
        "rubric": {
            "1": "Does the prompt clearly request translation?",
            "2": "Does it specify the source and target languages?",
            "3": "Does it mention meaning, tone, context, or naturalness?",
            "4": "Is the prompt complete enough to guide accurate translation?"
        },
        "focus": ["meaning", "tone", "context", "natural wording", "cultural meaning"],
        "formats": ["direct translation", "natural English sentence", "formal translation", "simple translation"],
        "audiences": ["English speakers", "students", "general readers", "beginners"],
        "constraints": ["avoid literal translation", "preserve the original tone", "keep the meaning accurate", "use fluent English"]
    },

    "Write a professional email": {
        "reference": "Write a polite professional email with a clear purpose, relevant context, and a specific requested action.",
        "verb": "write a professional email",
        "rubric": {
            "1": "Does the prompt clearly request an email?",
            "2": "Does it specify the email purpose and context?",
            "3": "Does it mention tone, structure, or requested action?",
            "4": "Is the prompt complete enough to produce a professional email?"
        },
        "focus": ["purpose", "context", "requested action", "polite tone", "clear subject"],
        "formats": ["formal email", "short email", "email with subject line", "structured email"],
        "audiences": ["a professor", "a manager", "a client", "a colleague"],
        "constraints": ["polite tone", "clear subject line", "short explanation", "respectful closing"]
    },

    "Generate social media captions": {
        "reference": "Generate engaging social media captions with a clear tone, target audience, and relevant hashtags.",
        "verb": "generate social media captions",
        "rubric": {
            "1": "Does the prompt clearly request captions?",
            "2": "Does it specify platform, topic, or audience?",
            "3": "Does it include tone, length, hashtags, or style constraints?",
            "4": "Is the prompt complete enough to generate suitable captions?"
        },
        "focus": ["engagement", "brand tone", "hashtags", "short wording", "target audience"],
        "formats": ["Instagram captions", "TikTok captions", "short captions", "caption list"],
        "audiences": ["young audience", "fashion followers", "students", "general audience"],
        "constraints": ["under 20 words", "include 3 hashtags", "friendly tone", "include emojis"]
    },

    "Explain machine learning concepts": {
        "reference": "Explain the machine learning concept in simple language using a practical example and avoiding unnecessary jargon.",
        "verb": "explain machine learning",
        "rubric": {
            "1": "Does the prompt clearly ask for an explanation?",
            "2": "Does it identify the topic or concept?",
            "3": "Does it specify audience level, examples, or simplicity?",
            "4": "Is the prompt complete enough to produce an understandable explanation?"
        },
        "focus": ["basic meaning", "real-world example", "step-by-step explanation", "simple analogy", "main idea"],
        "formats": ["short explanation", "bullet points", "step-by-step explanation", "simple paragraph"],
        "audiences": ["beginners", "first-year students", "non-technical readers", "computer science students"],
        "constraints": ["avoid jargon", "under 150 words", "use one example", "use simple language"]
    },

    "Write a customer support response": {
        "reference": "Write a polite customer support response that acknowledges the complaint, apologizes, explains the next step, and offers a solution.",
        "verb": "write a customer support response",
        "rubric": {
            "1": "Does the prompt clearly request a support response?",
            "2": "Does it describe the customer issue?",
            "3": "Does it mention empathy, apology, solution, or next step?",
            "4": "Is the prompt complete enough to produce a helpful professional reply?"
        },
        "focus": ["customer complaint", "apology", "solution", "next step", "empathy"],
        "formats": ["short reply", "professional response", "support email", "chat response"],
        "audiences": ["angry customer", "confused customer", "new customer", "returning customer"],
        "constraints": ["empathetic tone", "clear solution", "apologize politely", "include next steps"]
    },

    "Create study notes": {
        "reference": "Create concise study notes with key definitions, examples, and important points.",
        "verb": "create study notes",
        "rubric": {
            "1": "Does the prompt clearly request study notes?",
            "2": "Does it specify the topic or learning goal?",
            "3": "Does it include format, examples, or level of detail?",
            "4": "Is the prompt complete enough to produce useful notes?"
        },
        "focus": ["definitions", "key points", "examples", "exam review", "important terms"],
        "formats": ["bullet notes", "table format", "numbered notes", "short study guide"],
        "audiences": ["students", "beginners", "exam takers", "classmates"],
        "constraints": ["simple language", "include examples", "keep it concise", "highlight key terms"]
    },

    "Generate interview questions": {
        "reference": "Generate interview questions with clear difficulty levels and short expected answers.",
        "verb": "generate interview questions",
        "rubric": {
            "1": "Does the prompt clearly request interview questions?",
            "2": "Does it specify the topic or role?",
            "3": "Does it include number, difficulty, or answer format?",
            "4": "Is the prompt complete enough to produce useful interview questions?"
        },
        "focus": ["technical skills", "problem solving", "difficulty levels", "expected answers", "core concepts"],
        "formats": ["list of questions", "table", "numbered questions", "questions with answers"],
        "audiences": ["junior developers", "students", "job candidates", "beginners"],
        "constraints": ["generate 5 questions", "include answers", "increase difficulty", "keep questions clear"]
    },

    "Rewrite text formally": {
        "reference": "Rewrite the text in a formal tone while preserving the original meaning.",
        "verb": "rewrite the text formally",
        "rubric": {
            "1": "Does the prompt clearly request rewriting?",
            "2": "Does it specify the desired tone or style?",
            "3": "Does it mention preserving meaning or improving clarity?",
            "4": "Is the prompt complete enough to guide a high-quality rewrite?"
        },
        "focus": ["formal tone", "clarity", "professional wording", "original meaning", "grammar"],
        "formats": ["formal paragraph", "professional version", "polished rewrite", "clear rewrite"],
        "audiences": ["professors", "managers", "clients", "academic readers"],
        "constraints": ["preserve meaning", "improve clarity", "avoid slang", "keep it concise"]
    },

    "Create a presentation outline": {
        "reference": "Create a clear presentation outline with slide titles, key points, and logical flow.",
        "verb": "create a presentation outline",
        "rubric": {
            "1": "Does the prompt clearly request an outline?",
            "2": "Does it specify the topic or purpose?",
            "3": "Does it include slide count, structure, or audience?",
            "4": "Is the prompt complete enough to produce a useful presentation outline?"
        },
        "focus": ["slide titles", "main points", "logical flow", "introduction and conclusion", "audience"],
        "formats": ["slide outline", "numbered outline", "table", "bullet-point plan"],
        "audiences": ["students", "professors", "class audience", "beginners"],
        "constraints": ["6 slides", "include speaker notes", "clear structure", "short bullet points"]
    },

    "Write a product description": {
        "reference": "Write a persuasive product description that highlights features, benefits, and the target customer.",
        "verb": "write a product description",
        "rubric": {
            "1": "Does the prompt clearly request a product description?",
            "2": "Does it specify product features or target audience?",
            "3": "Does it mention tone, benefits, or format?",
            "4": "Is the prompt complete enough to create a persuasive description?"
        },
        "focus": ["main features", "customer benefits", "target audience", "selling points", "use cases"],
        "formats": ["short paragraph", "bullet list", "marketing copy", "online store description"],
        "audiences": ["online shoppers", "young adults", "students", "professionals"],
        "constraints": ["persuasive tone", "under 120 words", "mention benefits", "clear call to action"]
    },

    "Explain a scientific concept": {
        "reference": "Explain the scientific concept clearly using simple language, an example, and the main cause-effect relationship.",
        "verb": "explain the scientific concept",
        "rubric": {
            "1": "Does the prompt clearly request an explanation?",
            "2": "Does it specify the concept or scientific focus?",
            "3": "Does it include audience level, examples, or simplicity?",
            "4": "Is the prompt complete enough to produce a clear scientific explanation?"
        },
        "focus": ["cause and effect", "main definition", "real-world example", "key process", "simple analogy"],
        "formats": ["short paragraph", "bullet points", "step-by-step explanation", "beginner explanation"],
        "audiences": ["high school students", "beginners", "non-science readers", "university students"],
        "constraints": ["avoid complex jargon", "include one example", "under 150 words", "explain step by step"]
    },

    "Simplify complex text": {
        "reference": "Simplify the text while preserving the original meaning and making it easier for the target audience to understand.",
        "verb": "simplify the complex text",
        "rubric": {
            "1": "Does the prompt clearly request simplification?",
            "2": "Does it specify the target audience or reading level?",
            "3": "Does it mention preserving meaning or reducing jargon?",
            "4": "Is the prompt complete enough to produce a clear simplified version?"
        },
        "focus": ["main meaning", "simple wording", "important details", "clear explanation", "reduced jargon"],
        "formats": ["simple paragraph", "bullet points", "plain English version", "short explanation"],
        "audiences": ["beginners", "children", "non-experts", "students"],
        "constraints": ["preserve meaning", "avoid jargon", "use simple words", "keep it concise"]
    },

    "Correct grammar and spelling": {
        "reference": "Correct grammar, spelling, and punctuation while preserving the original meaning and style.",
        "verb": "correct grammar and spelling",
        "rubric": {
            "1": "Does the prompt clearly request correction?",
            "2": "Does it specify what should be corrected?",
            "3": "Does it mention preserving meaning or style?",
            "4": "Is the prompt complete enough to guide accurate editing?"
        },
        "focus": ["grammar", "spelling", "punctuation", "sentence clarity", "original meaning"],
        "formats": ["corrected paragraph", "edited version", "list of corrections", "clean final version"],
        "audiences": ["students", "writers", "professionals", "English learners"],
        "constraints": ["preserve meaning", "do not rewrite too much", "fix punctuation", "keep original tone"]
    },

    "Compare two concepts": {
        "reference": "Compare two concepts clearly by explaining similarities, differences, and practical examples.",
        "verb": "compare two concepts",
        "rubric": {
            "1": "Does the prompt clearly request comparison?",
            "2": "Does it specify the concepts being compared?",
            "3": "Does it mention similarities, differences, examples, or format?",
            "4": "Is the prompt complete enough to produce a useful comparison?"
        },
        "focus": ["similarities", "differences", "examples", "advantages and disadvantages", "main use cases"],
        "formats": ["comparison table", "bullet points", "short paragraph", "structured comparison"],
        "audiences": ["students", "beginners", "decision makers", "general readers"],
        "constraints": ["include examples", "use a table", "keep it concise", "explain simply"]
    }
}

SCORE_0_PROMPTS = [
    "Help me.",
    "Do this.",
    "Anything.",
    "Write something.",
    "I need it.",
    "Please.",
    "Make it good.",
    "Fix this.",
    "I don't know.",
    "Tell me about it.",
    "Can you?",
    "Need this fast.",
    "Whatever is fine.",
    "Just make it.",
    "Give me something.",
    "Do the thing.",
    "Make one.",
    "Work on it.",
    "This please.",
    "Complete it."
]

def choose(items):
    return random.choice(items)

def make_submission(task_data, score):
    verb = task_data["verb"]
    focus = choose(task_data["focus"])
    fmt = choose(task_data["formats"])
    audience = choose(task_data["audiences"])
    constraint = choose(task_data["constraints"])

    if score == 0:
        return choose(SCORE_0_PROMPTS)

    if score == 1:
        return choose([
            verb.capitalize() + ".",
            f"Can you {verb}?",
            f"Please {verb}.",
            f"I want you to {verb}.",
            f"Help me {verb}."
        ])

    if score == 2:
        return choose([
            f"Please {verb} clearly.",
            f"Can you {verb} with some details?",
            f"{verb.capitalize()} and include the main idea.",
            f"Help me {verb} in a simple way.",
            f"{verb.capitalize()} with a short explanation."
        ])

    if score == 3:
        return choose([
            f"Please {verb} for {audience} and focus on {focus}.",
            f"{verb.capitalize()} using {fmt} and include {focus}.",
            f"Can you {verb} clearly with {focus} and {constraint}?",
            f"{verb.capitalize()} in a useful way for {audience}.",
            f"Please {verb} with {fmt} and make it suitable for {audience}."
        ])

    return choose([
        f"Please {verb} for {audience} using {fmt}, focusing on {focus}, and make sure to {constraint}.",
        f"{verb.capitalize()} with {fmt}, include {focus}, use a style suitable for {audience}, and {constraint}.",
        f"Create a high-quality response that will {verb}, target {audience}, include {focus}, and follow this constraint: {constraint}.",
        f"{verb.capitalize()} in a complete and clear way for {audience}; use {fmt}, cover {focus}, and {constraint}.",
        f"Please {verb} with a clear structure, target {audience}, include {focus}, use {fmt}, and {constraint}."
    ])

def make_rationale(score, task, submission):
    if score == 0:
        return (
            f"The prompt '{submission}' is not usable for the task '{task}'. "
            f"It does not clearly state what output is needed, so it fails the clarity and completeness criteria."
        )

    if score == 1:
        return (
            f"The prompt '{submission}' is related to the task, but it is too vague. "
            f"It does not specify format, audience, focus, or constraints, so the model would have little guidance."
        )

    if score == 2:
        return (
            f"The prompt '{submission}' communicates the general intention, but it is incomplete. "
            f"It gives some direction but misses important rubric elements such as specific focus, output format, or constraints."
        )

    if score == 3:
        return (
            f"The prompt '{submission}' is clear and mostly useful for the task. "
            f"It satisfies clarity and some specificity, but it could be improved with more complete constraints or expected output details."
        )

    return (
        f"The prompt '{submission}' is strong because it clearly defines the task, audience, focus, format, and constraints. "
        f"It satisfies the rubric well and would likely produce a high-quality response."
    )

def make_entry(task_name, score):
    task_data = TASKS[task_name]
    submission = make_submission(task_data, score)

    return {
        "task": task_name,
        "reference": task_data["reference"],
        "submission": submission,
        "rubric": task_data["rubric"],
        "score": score,
        "rationale": make_rationale(score, task_name, submission)
    }

def main():
    dataset = []
    seen = set()
    score_counts = Counter()

    attempts = 0
    max_attempts = 500000

    while min([score_counts[s] for s in range(5)]) < TARGET_PER_SCORE:
        attempts += 1

        if attempts > max_attempts:
            print("Stopped early. Not enough unique examples.")
            break

        score = random.choice([0, 1, 2, 3, 4])

        if score_counts[score] >= TARGET_PER_SCORE:
            continue

        task_name = random.choice(list(TASKS.keys()))
        entry = make_entry(task_name, score)

        key = (
            entry["task"].lower().strip(),
            entry["reference"].lower().strip(),
            entry["submission"].lower().strip()
        )

        if key in seen:
            continue

        seen.add(key)
        dataset.append(entry)
        score_counts[score] += 1

    random.shuffle(dataset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Total samples: {len(dataset)}")
    print("Score distribution:", dict(score_counts))

if __name__ == "__main__":
    main()

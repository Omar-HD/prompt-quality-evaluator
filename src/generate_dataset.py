import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

OUTPUT_PATH = Path("data/raw/prompt_quality_dataset.json")

# 100,000 total = 20,000 per score
TARGET_PER_SCORE = 20000

TASKS = [
    ("Summarize an article", "summarize the article", "Summarize the article clearly by focusing on the main idea, key arguments, and conclusion."),
    ("Translate Arabic to English", "translate the Arabic text into English", "Translate the Arabic text into fluent English while preserving meaning, tone, and context."),
    ("Write a professional email", "write a professional email", "Write a polite professional email with a clear purpose, relevant context, and a specific requested action."),
    ("Generate social media captions", "generate social media captions", "Generate engaging social media captions with a clear tone, target audience, and relevant hashtags."),
    ("Explain machine learning concepts", "explain machine learning", "Explain the machine learning concept in simple language using a practical example and avoiding unnecessary jargon."),
    ("Write a customer support response", "write a customer support response", "Write a polite customer support response that acknowledges the complaint, apologizes, explains the next step, and offers a solution."),
    ("Create study notes", "create study notes", "Create concise study notes with key definitions, examples, and important points."),
    ("Generate interview questions", "generate interview questions", "Generate interview questions with clear difficulty levels and short expected answers."),
    ("Rewrite text formally", "rewrite the text formally", "Rewrite the text in a formal tone while preserving the original meaning."),
    ("Create a presentation outline", "create a presentation outline", "Create a clear presentation outline with slide titles, key points, and logical flow."),
    ("Write a product description", "write a product description", "Write a persuasive product description that highlights features, benefits, and the target customer."),
    ("Explain a scientific concept", "explain the scientific concept", "Explain the scientific concept clearly using simple language, an example, and the main cause-effect relationship."),
    ("Simplify complex text", "simplify the complex text", "Simplify the text while preserving the original meaning and making it easier for the target audience to understand."),
    ("Correct grammar and spelling", "correct grammar and spelling", "Correct grammar, spelling, and punctuation while preserving the original meaning and style."),
    ("Compare two concepts", "compare two concepts", "Compare two concepts clearly by explaining similarities, differences, and practical examples."),
    ("Extract key information", "extract key information from the text", "Extract the most important information from the text in a clear structured format."),
    ("Classify text topic", "classify the text topic", "Classify the text into the most appropriate topic category and explain the reason briefly."),
    ("Analyze sentiment", "analyze the sentiment of the text", "Analyze the sentiment of the text and identify whether it is positive, negative, or neutral."),
    ("Write a project abstract", "write a project abstract", "Write a concise project abstract that explains the problem, method, and expected result."),
    ("Generate quiz questions", "generate quiz questions", "Generate quiz questions with correct answers and clear difficulty levels."),
    ("Create a lesson plan", "create a lesson plan", "Create a structured lesson plan with objectives, activities, and assessment methods."),
    ("Write a bug report", "write a bug report", "Write a clear bug report with steps to reproduce, expected behavior, and actual behavior."),
    ("Review a resume bullet", "review a resume bullet", "Evaluate and improve a resume bullet to make it more specific, measurable, and professional."),
    ("Write a meeting agenda", "write a meeting agenda", "Write a clear meeting agenda with topics, timing, and expected outcomes."),
    ("Generate SQL query explanation", "explain an SQL query", "Explain what an SQL query does using simple language and examples.")
]

RUBRIC_DEFAULT = {
    "1": "Does the prompt clearly state the task?",
    "2": "Does it include specific requirements?",
    "3": "Does it specify useful constraints such as format, audience, tone, or length?",
    "4": "Is the prompt complete and useful enough to guide a high-quality response?"
}

TOPICS = [
    "climate change", "online learning", "customer delivery delay", "machine learning", "data privacy",
    "healthy habits", "software engineering", "Arabic translation", "university registration",
    "job interview preparation", "product marketing", "social media campaign", "scientific method",
    "neural networks", "remote work", "time management", "mobile app design", "cybersecurity",
    "exam revision", "business proposal", "medical appointment", "refund request", "technical support issue",
    "course project", "environmental awareness", "AI ethics", "sports event", "travel plan",
    "restaurant review", "book chapter", "research paper", "lecture notes", "shopping product",
    "team meeting", "internship application", "grammar editing", "formal complaint", "presentation slides",
    "student assignment", "project report", "meeting reminder", "science experiment", "data analysis",
    "programming basics", "cloud computing", "NLP project", "database design", "customer feedback",
    "marketing campaign", "study schedule", "public speaking", "product launch", "SQL databases",
    "bug fixing", "resume writing", "lesson planning", "quiz preparation", "topic classification"
]

FOCUS = [
    "main idea", "key arguments", "supporting evidence", "conclusion", "important details",
    "audience needs", "tone", "structure", "clarity", "main problem", "core concept",
    "examples", "limitations", "advantages", "disadvantages", "next steps", "expected output",
    "reader understanding", "context", "accuracy", "professional wording", "practical use",
    "evaluation criteria", "root cause", "correctness", "measurable impact", "learning objective"
]

FORMATS = [
    "bullet points", "numbered list", "short paragraph", "table", "one paragraph", "Q&A format",
    "checklist", "mini report", "study-note format", "formal paragraph", "short answer",
    "structured outline", "comparison table", "clean final version", "brief explanation",
    "step-by-step format", "two-column table", "executive summary", "slide outline",
    "email format", "chat reply", "caption list", "JSON format", "markdown table", "rubric format"
]

AUDIENCES = [
    "beginners", "students", "university students", "high school students", "general readers",
    "busy readers", "non-experts", "professionals", "a professor", "a manager", "a client",
    "classmates", "English learners", "job candidates", "technical readers", "children",
    "small business owners", "online shoppers", "social media followers", "new learners",
    "first-year computer science students", "customers", "teachers", "project reviewers",
    "software testers", "resume reviewers", "meeting participants", "data analysts"
]

CONSTRAINTS = [
    "under 50 words", "under 100 words", "under 150 words", "in 5 bullet points",
    "using simple language", "with a polite tone", "with a formal tone", "with examples",
    "without adding outside information", "with a clear subject line", "with 3 hashtags",
    "with one real-world example", "avoiding jargon", "preserving the original meaning",
    "keeping the response concise", "including a short summary", "including a clear next step",
    "using neutral tone", "including similarities and differences", "ending with a recommendation",
    "including expected answers", "using a beginner-friendly style", "with no more than 3 sentences",
    "including a clear structure", "mentioning the target audience", "using professional wording",
    "including measurable details", "explaining the reason", "using valid JSON", "listing assumptions"
]

SCENARIOS = [
    "for a university assignment", "for a class presentation", "for a customer message",
    "for a business report", "for a social media post", "for an exam review",
    "for a job application", "for a project demo", "for a professor", "for a client",
    "for a beginner tutorial", "for a team meeting", "for a product page", "for a research summary",
    "for an online course", "for a study group", "for a technical explanation", "for a public announcement",
    "for a short video script", "for a professional conversation", "for an academic discussion",
    "for a support ticket", "for a marketing campaign", "for a comparison task", "for a GitHub README",
    "for a resume review", "for a bug tracker", "for a lesson session", "for a quiz bank"
]

VAGUE_ZERO = [
    "Help me", "Do this", "Anything is fine", "Write something", "I need it", "Please",
    "Make it good", "Fix this", "I do not know", "Tell me about it", "Can you",
    "Need this fast", "Whatever works", "Just make it", "Give me something",
    "Do the thing", "Make one", "Work on it", "This please", "Complete it",
    "Make it better", "Handle this", "Use this", "Do something useful", "I need help"
]

RATIONALE_OPENERS = {
    0: ["This is not a usable prompt", "This prompt is too incomplete to evaluate properly", "The submission gives almost no task guidance", "The prompt fails to communicate a meaningful request"],
    1: ["This prompt is connected to the task but remains very vague", "The task is present, but the prompt gives minimal guidance", "This is a weak prompt because it only states the broad request", "The prompt has a recognizable goal but lacks useful detail"],
    2: ["This prompt communicates the general intention", "The prompt is understandable but still incomplete", "This is an acceptable starting point, but it lacks important details", "The prompt gives partial guidance but remains under-specified"],
    3: ["This is a good prompt with useful guidance", "The prompt is mostly clear and relevant", "This prompt provides a helpful direction for the task", "The prompt satisfies several rubric criteria but is not fully controlled"],
    4: ["This is a strong prompt", "This prompt is highly effective", "The prompt gives complete and well-controlled instructions", "The prompt is clear, specific, and task-appropriate"]
}

def choose(items):
    return random.choice(items)

def cap(text):
    return text[:1].upper() + text[1:]

def make_submission(verb, score):
    topic = choose(TOPICS)
    focus = choose(FOCUS)
    fmt = choose(FORMATS)
    audience = choose(AUDIENCES)
    constraint = choose(CONSTRAINTS)
    scenario = choose(SCENARIOS)

    if score == 0:
        return choose([
            f"{choose(VAGUE_ZERO)}.",
            f"{choose(VAGUE_ZERO)} {scenario}.",
            f"{choose(VAGUE_ZERO)} about {topic}.",
            f"{choose(VAGUE_ZERO)} please, it is urgent.",
            f"{choose(VAGUE_ZERO)} and make it nice.",
            f"{choose(VAGUE_ZERO)} with this {topic} thing.",
            f"{choose(VAGUE_ZERO)} {scenario} about {topic}."
        ])

    if score == 1:
        return choose([
            f"{cap(verb)}.",
            f"Can you {verb}?",
            f"Please {verb}.",
            f"I want you to {verb}.",
            f"Help me {verb}.",
            f"{cap(verb)} for me.",
            f"{cap(verb)} quickly.",
            f"{cap(verb)} clearly.",
            f"I need you to {verb}.",
            f"Can you help me {verb}?",
            f"{cap(verb)} about {topic}.",
            f"{cap(verb)} {scenario}.",
            f"Do a basic version to {verb} about {topic}.",
            f"Give me something to {verb} {scenario}.",
            f"Just {verb}."
        ])

    if score == 2:
        return choose([
            f"Please {verb} clearly about {topic}.",
            f"Can you {verb} with some details?",
            f"{cap(verb)} and include the main idea.",
            f"Help me {verb} in a simple way {scenario}.",
            f"{cap(verb)} with a short explanation.",
            f"{cap(verb)} for {audience}.",
            f"{cap(verb)} and focus on {focus}.",
            f"Please {verb} using {fmt}.",
            f"{cap(verb)} while keeping it concise.",
            f"{cap(verb)} and make it easy to understand.",
            f"{cap(verb)} using clear language.",
            f"{cap(verb)} and organize the response well.",
            f"{cap(verb)} with examples if possible.",
            f"{cap(verb)} about {topic} and include some useful details.",
            f"Use {fmt} to {verb} {scenario}."
        ])

    if score == 3:
        return choose([
            f"Please {verb} for {audience} and focus on {focus}.",
            f"{cap(verb)} using {fmt} and include {focus}.",
            f"Can you {verb} clearly with {focus} and {constraint}?",
            f"{cap(verb)} in a useful way for {audience}.",
            f"Please {verb} with {fmt} and make it suitable for {audience}.",
            f"{cap(verb)} with a clear structure and focus on {focus}.",
            f"Please {verb} for {audience}, using clear language and {fmt}.",
            f"{cap(verb)} with {constraint} and include {focus}.",
            f"{cap(verb)} about {topic} for {audience} and focus on {focus}.",
            f"Please {verb} with a clear structure {scenario}."
        ])

    return choose([
        f"Please {verb} about {topic} for {audience} using {fmt}, focusing on {focus}, and make sure to {constraint}.",
        f"{cap(verb)} with {fmt}, include {focus}, use a style suitable for {audience}, and {constraint}.",
        f"Create a high-quality response that will {verb}, target {audience}, include {focus}, and follow this constraint: {constraint}.",
        f"{cap(verb)} in a complete and clear way for {audience}; use {fmt}, cover {focus}, and {constraint}.",
        f"Please {verb} with a clear structure, target {audience}, include {focus}, use {fmt}, and {constraint}.",
        f"Produce a polished response to {verb}; use {fmt}, focus on {focus}, target {audience}, and {constraint}.",
        f"Create a detailed and controlled response to {verb}; topic: {topic}; audience: {audience}; format: {fmt}; focus: {focus}; constraint: {constraint}.",
        f"Please {verb} in a well-organized {fmt}, suitable for {audience}, covering {focus}, and following this requirement: {constraint}."
    ])

def make_rationale(score, task_name, submission):
    opener = choose(RATIONALE_OPENERS[score])
    if score == 0:
        return f"{opener}. The submission '{submission}' does not clearly connect to the task '{task_name}' and fails the rubric on clarity, specificity, and completeness."
    if score == 1:
        return f"{opener}. The submission '{submission}' mentions the general task, but it does not specify the expected format, audience, focus, or constraints."
    if score == 2:
        return f"{opener}. The submission '{submission}' gives some direction, but it only partially satisfies the rubric because key details such as focus, format, audience, or constraints are missing."
    if score == 3:
        return f"{opener}. The submission '{submission}' satisfies clarity and some specificity, but it could be improved by adding more complete constraints or a more precise output format."
    return f"{opener}. The submission '{submission}' clearly defines the task, includes relevant constraints, and gives enough detail about audience, focus, or format to satisfy the rubric well."

def main():
    dataset = []
    score_counts = Counter()

    for score in range(5):
        for _ in range(TARGET_PER_SCORE):
            task_name, verb, reference = choose(TASKS)
            submission = make_submission(verb, score)
            dataset.append({
                "task": task_name,
                "reference": reference,
                "submission": submission,
                "rubric": RUBRIC_DEFAULT,
                "score": score,
                "rationale": make_rationale(score, task_name, submission)
            })
            score_counts[score] += 1

    random.shuffle(dataset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Total samples: {len(dataset)}")
    print("Score distribution:", dict(score_counts))

if __name__ == "__main__":
    main()

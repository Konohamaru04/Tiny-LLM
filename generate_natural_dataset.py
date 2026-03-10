import os
import json
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "data" / "raw"
SFT_DIR = BASE_DIR / "data" / "sft"

RAW_DIR.mkdir(parents=True, exist_ok=True)
SFT_DIR.mkdir(parents=True, exist_ok=True)

topics = [
    {
        "name": "machine learning",
        "definition": "the practice of training models to learn patterns from data",
        "uses": ["spam detection", "forecasting", "classification", "recommendation systems"],
        "pitfalls": ["overfitting", "data leakage", "poor validation", "weak feature quality"],
    },
    {
        "name": "deep learning",
        "definition": "a subset of machine learning built around multi-layer neural networks",
        "uses": ["image recognition", "speech processing", "text generation", "representation learning"],
        "pitfalls": ["high compute cost", "overfitting", "unstable training", "data hunger"],
    },
    {
        "name": "transformers",
        "definition": "neural network architectures that rely heavily on attention mechanisms",
        "uses": ["language modeling", "translation", "summarization", "question answering"],
        "pitfalls": ["memory cost", "long-context inefficiency", "hallucinated confidence", "expensive training"],
    },
    {
        "name": "tokenization",
        "definition": "the process of splitting text into smaller units a model can process",
        "uses": ["language modeling", "text preprocessing", "subword segmentation", "prompt handling"],
        "pitfalls": ["fragmented text", "vocabulary mismatch", "excess token count", "special token misuse"],
    },
    {
        "name": "PyTorch",
        "definition": "a Python deep learning framework used to build and train neural networks",
        "uses": ["research prototyping", "model training", "GPU computation", "custom architectures"],
        "pitfalls": ["shape bugs", "device mismatch", "OOM errors", "silent dtype issues"],
    },
    {
        "name": "gradient descent",
        "definition": "an optimization process that updates parameters to reduce loss",
        "uses": ["training neural networks", "fitting regression models", "fine-tuning", "iterative optimization"],
        "pitfalls": ["bad learning rates", "slow convergence", "poor local minima", "unstable updates"],
    },
    {
        "name": "overfitting",
        "definition": "a condition where a model learns the training data too narrowly",
        "uses": ["diagnosing training problems", "regularization analysis", "validation monitoring", "model selection"],
        "pitfalls": ["misleading accuracy", "weak generalization", "memorization", "fragile predictions"],
    },
    {
        "name": "regularization",
        "definition": "a set of techniques used to improve generalization and reduce overfitting",
        "uses": ["dropout", "weight decay", "early stopping", "simpler architectures"],
        "pitfalls": ["underfitting", "over-penalizing", "blunt hyperparameter choices", "misread validation trends"],
    },
    {
        "name": "embeddings",
        "definition": "dense vector representations of discrete items such as words or tokens",
        "uses": ["semantic similarity", "retrieval", "language modeling", "feature learning"],
        "pitfalls": ["poor coverage", "domain mismatch", "semantic drift", "insufficient context"],
    },
    {
        "name": "attention",
        "definition": "a mechanism that lets a model focus on the most relevant parts of an input",
        "uses": ["sequence modeling", "transformers", "alignment", "context selection"],
        "pitfalls": ["high memory use", "spurious focus", "confusing interpretation", "costly scaling"],
    },
    {
        "name": "dropout",
        "definition": "a regularization method that randomly disables parts of a network during training",
        "uses": ["generalization improvement", "small-data training", "preventing co-adaptation", "stabilizing training"],
        "pitfalls": ["too much noise", "underfitting", "misconfigured rates", "wrong expectations at inference"],
    },
    {
        "name": "validation sets",
        "definition": "held-out data used to estimate how well a model generalizes",
        "uses": ["early stopping", "hyperparameter tuning", "checkpoint selection", "progress monitoring"],
        "pitfalls": ["leakage", "too small a split", "distribution mismatch", "overusing the validation set"],
    },
]

styles = [
    "engineering note",
    "practical guide",
    "internal memo",
    "tutorial outline",
    "plain-English explainer",
    "team documentation page",
    "FAQ page",
    "short blog post",
    "troubleshooting note",
    "reference note",
]

openers = [
    "People often first encounter {topic} when they are trying to make a system behave less like a calculator and more like a learner.",
    "At a high level, {topic} sounds simple until you try to build something real with it.",
    "The easiest way to understand {topic} is to stop treating it like abstract theory and start viewing it as an engineering tool.",
    "In practice, {topic} becomes much clearer once you see how it behaves under constraints like limited data or limited compute.",
    "A lot of confusion around {topic} comes from explanations that are technically correct but not especially usable.",
]

closers = [
    "That is why small, repeatable experiments are often more educational than large but vague ones.",
    "In the end, the best way to learn it is to build a small version and inspect every step.",
    "The concept gets easier once you connect the math, the code, and the failure cases.",
    "Good results usually come from discipline in data, evaluation, and iteration rather than from hype.",
    "A simple system that you understand is often more valuable than a larger one you cannot debug.",
]

section_templates = [
    ["Why it matters", "Common mistakes", "A small example", "Practical takeaway"],
    ["What it is", "Where it helps", "What usually goes wrong", "Final note"],
    ["Quick definition", "Real-world use", "Failure modes", "Checklist"],
    ["Overview", "Engineering intuition", "Warnings", "Summary"],
]

def pick(seq):
    return random.choice(seq)

def sentence_variants(topic):
    name = topic["name"]
    definition = topic["definition"]
    use = pick(topic["uses"])
    pitfall = pick(topic["pitfalls"])
    return [
        f"{name.capitalize()} can be described as {definition}.",
        f"A common use case for {name} is {use}.",
        f"One reason teams adopt {name} is that it can simplify difficult modeling problems when used carefully.",
        f"At the same time, {pitfall} is a frequent source of confusion.",
        f"In smaller projects, the value of {name} often comes from clarity and reproducibility rather than raw scale.",
        f"When someone is learning {name}, the most useful habit is usually to inspect inputs, outputs, and assumptions step by step.",
        f"Even a modest implementation can teach a lot if the pipeline is clean and the evaluation is honest.",
    ]

def make_markdown_doc(index, topic):
    style = pick(styles)
    title = f"{topic['name'].title()} - {style.title()} #{index+1}"
    sections = pick(section_templates)
    sents = sentence_variants(topic)
    random.shuffle(sents)

    intro = pick(openers).format(topic=topic["name"])
    close = pick(closers)

    body = [
        f"# {title}",
        "",
        "## Introduction",
        "",
        intro,
        " " + pick(sents),
        "",
    ]

    for section_name in sections:
        body.extend([
            f"## {section_name}",
            "",
        ])

        paragraph_count = random.randint(1, 2)
        for _ in range(paragraph_count):
            local = random.sample(sents, k=min(len(sents), random.randint(2, 4)))
            paragraph = " ".join(local)
            body.append(paragraph)
            body.append("")

        if random.random() < 0.55:
            bullets = random.sample(
                [
                    f"Prefer small experiments before scaling up {topic['name']}.",
                    f"Track validation behavior rather than trusting training loss alone.",
                    f"Write down assumptions before changing hyperparameters.",
                    f"Look for simple baselines before reaching for complex fixes.",
                    f"Keep the data pipeline understandable.",
                    f"Document what failed, not just what worked.",
                ],
                k=3,
            )
            for item in bullets:
                body.append(f"- {item}")
            body.append("")

    if random.random() < 0.45:
        body.extend([
            "## Quick Example",
            "",
            f"Suppose a small team is experimenting with {topic['name']} on a limited local dataset.",
            f"They start with a simple baseline, monitor validation behavior, and only then increase complexity.",
            "That sequence usually reveals more than an oversized first attempt.",
            "",
        ])

    body.extend([
        "## Closing Thoughts",
        "",
        close,
        "",
    ])

    return "\n".join(body).strip() + "\n"

def build_markdown_files(count=100):
    for i in range(count):
        topic = topics[i % len(topics)]
        doc = make_markdown_doc(i, topic)
        filename = RAW_DIR / f"{i+1:03d}_{topic['name'].replace(' ', '_')}.md"
        filename.write_text(doc, encoding="utf-8")

systems = [
    "You are a concise and helpful assistant.",
    "You are a practical technical assistant who explains things clearly.",
    "You are a careful assistant who avoids unnecessary jargon.",
    "You are a helpful assistant for software, machine learning, and local model development.",
    "You are a calm assistant who gives direct answers first and detail second.",
]

user_prompts = [
    "What is {topic}?",
    "Explain {topic} in simple terms.",
    "Why is {topic} important?",
    "How would you explain {topic} to a beginner?",
    "What is the practical role of {topic}?",
    "Can you give me a short explanation of {topic}?",
    "What usually goes wrong with {topic}?",
    "When should someone use {topic}?",
]

answer_openers = [
    "{topic_cap} is {definition}.",
    "In simple terms, {topic} is {definition}.",
    "A practical way to think about {topic} is this: it is {definition}.",
]

answer_followups = [
    "It becomes useful when a system needs to learn patterns, make better predictions, or handle complex data.",
    "In real projects, its value usually depends on data quality, evaluation discipline, and sensible defaults.",
    "People often misunderstand it by focusing only on theory and ignoring failure cases.",
    "For beginners, the best path is usually to start small and inspect each stage of the pipeline.",
    "It is easier to learn when you connect the idea to one concrete example instead of ten abstract definitions.",
]

json_tasks = [
    (
        "Return a JSON object with keys task, priority, and done for a task named 'train tokenizer', priority 'high', done false.",
        '<|json|>\n{"task": "train tokenizer", "priority": "high", "done": false}\n</json>',
    ),
    (
        "Return a JSON object with keys library and purpose for PyTorch.",
        '<|json|>\n{"library": "PyTorch", "purpose": "deep learning framework"}\n</json>',
    ),
    (
        "Return a JSON object with keys concept and note for overfitting.",
        '<|json|>\n{"concept": "overfitting", "note": "model memorizes training data and generalizes poorly"}\n</json>',
    ),
]

def build_sft_record(topic):
    topic_name = topic["name"]
    definition = topic["definition"]
    prompt = pick(user_prompts).format(topic=topic_name)

    answer_parts = [
        pick(answer_openers).format(topic=topic_name, topic_cap=topic_name.capitalize(), definition=definition),
        pick(answer_followups),
    ]

    if random.random() < 0.55:
        answer_parts.append(f"A common use case is {pick(topic['uses'])}.")
    if random.random() < 0.35:
        answer_parts.append(f"A frequent pitfall is {pick(topic['pitfalls'])}.")

    assistant = " ".join(answer_parts)

    return {
        "system": pick(systems),
        "user": prompt,
        "assistant": assistant,
    }

def build_sft_files(train_count=1000, val_count=120):
    train_records = []
    val_records = []

    for i in range(train_count):
        if i % 25 == 0:
            task_prompt, task_answer = pick(json_tasks)
            train_records.append({
                "system": "You are a helpful assistant. When the user asks for structured output, wrap it inside <|json|> and </json>.",
                "user": task_prompt,
                "assistant": task_answer,
            })
        else:
            train_records.append(build_sft_record(topics[i % len(topics)]))

    for i in range(val_count):
        val_records.append(build_sft_record(topics[(i * 3) % len(topics)]))

    train_path = SFT_DIR / "sample_train.jsonl"
    val_path = SFT_DIR / "sample_val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for row in train_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for row in val_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    build_markdown_files(count=100)
    build_sft_files(train_count=1000, val_count=120)
    print("Done.")
    print(f"Markdown files created in: {RAW_DIR}")
    print(f"SFT train file: {SFT_DIR / 'sample_train.jsonl'}")
    print(f"SFT val file:   {SFT_DIR / 'sample_val.jsonl'}")

if __name__ == "__main__":
    main()
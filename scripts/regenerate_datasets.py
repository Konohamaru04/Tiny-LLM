from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

import aiofiles
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SFT_DIR = ROOT / "data" / "sft"

SCENARIOS = [
    {
        "key": "support_triage",
        "setting": "a subscription software company where weekend tickets pile up",
        "role": "support lead",
        "goal": "separate urgent billing issues from routine product questions quickly",
        "pain": "calm but urgent cases keep getting buried under long feature requests",
        "artifact": "a CSV export from the help desk",
        "tool": "Python and Google Sheets",
        "metric": "median first response time",
        "constraint": "only two agents cover the weekend shift",
        "stakeholder": "the head of customer success",
        "task": "group tickets by age, tag, and channel so the on-call agent knows what to answer first",
        "fields": ("ticket_id", "tag", "channel", "created_at", "hours_open"),
        "notes": (
            "Invoice complaints often look polite even when they are time-sensitive.",
            "Some tags come from old automations and are no longer trustworthy.",
            "A small change in ordering would help more than a full system rewrite.",
        ),
    },
    {
        "key": "llm_training",
        "setting": "a small engineering team training a local language model on one shared GPU",
        "role": "ML engineer",
        "goal": "make training runs easier to debug without wasting days on guesswork",
        "pain": "people keep changing several variables at once and then argue about what helped",
        "artifact": "run logs, config files, and checkpoint notes",
        "tool": "PyTorch and YAML configs",
        "metric": "validation loss plus useful sample generations",
        "constraint": "the team only gets one serious overnight run each day",
        "stakeholder": "the teammate responsible for model quality",
        "task": "compare runs, surface the metrics that matter, and flag checkpoints worth testing in chat",
        "fields": ("run_name", "step", "train_loss", "val_loss", "json_valid_rate"),
        "notes": (
            "One checkpoint looked better by loss but produced flatter answers.",
            "Another run had slightly worse perplexity but clearly better chat behavior.",
            "Weak experiment notes have caused more confusion than weak hardware.",
        ),
    },
    {
        "key": "cafe_inventory",
        "setting": "an independent cafe that keeps running out of oat milk during the busiest hour",
        "role": "operations manager",
        "goal": "reorder just enough stock without turning the back room into storage overflow",
        "pain": "every shift lead estimates demand differently and nobody fully trusts the notes",
        "artifact": "a weekly sales sheet and a supplier note",
        "tool": "a spreadsheet and a reorder script",
        "metric": "stockout rate by item and hour",
        "constraint": "deliveries only arrive three mornings per week",
        "stakeholder": "the owner watching cash flow closely",
        "task": "summarize hourly sales and flag items likely to run out before the next delivery",
        "fields": ("date", "hour", "item", "cups_sold", "on_hand"),
        "notes": (
            "Cold drinks spike on warm days while pastry demand stays concentrated early.",
            "Baristas substitute milk on the fly and do not always log the swap.",
            "A late Friday delivery changes the whole weekend picture.",
        ),
    },
    {
        "key": "school_attendance",
        "setting": "a district office trying to catch attendance issues before families receive formal warnings",
        "role": "school operations analyst",
        "goal": "spot patterns early enough for a counselor to intervene gently",
        "pain": "by the time the weekly report is reviewed, the students who need help most are already further behind",
        "artifact": "an attendance export with teacher notes",
        "tool": "SQL and a dashboard",
        "metric": "students with repeated unexcused absences",
        "constraint": "privacy rules limit how much raw detail can be forwarded",
        "stakeholder": "the district attendance coordinator",
        "task": "filter repeated absence patterns and prepare a counselor-friendly summary without exposing private notes",
        "fields": ("student_id", "date", "status", "periods_missed", "note"),
        "notes": (
            "Missing first period every Monday may point to transportation trouble rather than motivation.",
            "Teacher notes are useful but inconsistent and not safe to forward verbatim.",
            "The district wants fewer surprise warnings and more timely human follow-up.",
        ),
    },
    {
        "key": "clinic_scheduling",
        "setting": "a community clinic that loses appointment slots when reminder calls happen too late",
        "role": "front-desk supervisor",
        "goal": "reduce no-shows without making patients feel nagged",
        "pain": "the reminder list is exported too late for same-week openings to be reused",
        "artifact": "an appointment roster and a cancellation log",
        "tool": "a scheduling export and a call script",
        "metric": "no-show rate by visit type",
        "constraint": "staff only have ninety minutes each afternoon for outreach",
        "stakeholder": "the clinic manager",
        "task": "identify the appointments most likely to no-show so the call list starts with the highest-value follow-up",
        "fields": ("patient_id", "visit_type", "appointment_time", "previous_no_shows", "contact_status"),
        "notes": (
            "Long specialty visits are harder to refill once they open up.",
            "Some patients answer texts far more reliably than phone calls.",
            "The team wants a practical call order, not another theoretical score.",
        ),
    },
    {
        "key": "warehouse_picking",
        "setting": "a regional warehouse where small picking errors create expensive reshipments",
        "role": "floor supervisor",
        "goal": "reduce avoidable mistakes without slowing the team to a crawl",
        "pain": "rush orders move fast, but their labels change more often than shelf habits do",
        "artifact": "scan logs and reshipment notes",
        "tool": "barcode exports and a short Python script",
        "metric": "mis-pick rate",
        "constraint": "night shift staffing changes every week",
        "stakeholder": "the logistics director",
        "task": "match scan history to reshipments so the supervisor can see where accuracy breaks down",
        "fields": ("order_id", "sku", "picker", "scan_time", "bin_location"),
        "notes": (
            "Temporary staff do better when the bin labels are clean and obvious.",
            "Most mistakes happen where packaging looks nearly identical.",
            "The director wants patterns and fixes, not blame language.",
        ),
    },
    {
        "key": "design_revisions",
        "setting": "a freelance design studio juggling client revisions across email, chat, and annotated PDFs",
        "role": "studio owner",
        "goal": "keep projects moving without losing the logic behind each design decision",
        "pain": "feedback arrives in fragments, and the same request gets rephrased three different ways",
        "artifact": "client comments and version notes",
        "tool": "Notion and a tracking table",
        "metric": "time from feedback to approved revision",
        "constraint": "the designer is also the project manager",
        "stakeholder": "a long-term client contact",
        "task": "normalize revision requests into one scoped list for the next design round",
        "fields": ("project", "source", "comment", "page", "status"),
        "notes": (
            "Clients say a page feels busy when they often mean the hierarchy is unclear.",
            "Pushback works better when it protects the original brief instead of sounding defensive.",
            "A clean recap saves more time than a faster first draft.",
        ),
    },
    {
        "key": "returns_analysis",
        "setting": "an online home-goods store trying to understand why returns jumped after a packaging change",
        "role": "ecommerce analyst",
        "goal": "separate actual defects from shipping damage and buyer mismatch",
        "pain": "notes from support, warehouse, and carrier systems do not line up cleanly",
        "artifact": "returns records and carrier incident notes",
        "tool": "SQL, spreadsheets, and a dashboard",
        "metric": "return rate by reason code",
        "constraint": "leadership wants an answer before the next packaging order goes out",
        "stakeholder": "the operations director",
        "task": "join order, return, and carrier tables to show which return reasons moved most after the packaging change",
        "fields": ("order_id", "sku", "reason_code", "carrier_flag", "package_version"),
        "notes": (
            "Shipping damage can look like a product defect when the photos are cropped badly.",
            "Customers describe fit and color issues differently from internal reason codes.",
            "A credible answer has to separate signal from messy labeling.",
        ),
    },
    {
        "key": "onboarding",
        "setting": "a growing software team where new hires lose time because setup knowledge lives in private chats",
        "role": "engineering manager",
        "goal": "help new teammates become useful in their first week without drowning them in documents",
        "pain": "every experienced engineer explains environment setup in a different order",
        "artifact": "setup notes, access requests, and starter tasks",
        "tool": "Markdown docs and a checklist app",
        "metric": "time to first merged pull request",
        "constraint": "the team supports multiple operating systems and local dev stacks",
        "stakeholder": "the newest engineer on the team",
        "task": "turn loose setup notes into a checklist that can be filtered by role and operating system",
        "fields": ("step_id", "team", "os", "task", "owner"),
        "notes": (
            "The biggest blocker is usually one missing permission buried in an old message.",
            "A clear first-week path lowers anxiety more than a giant knowledge base does.",
            "Good onboarding explains why a step matters, not just where to click.",
        ),
    },
    {
        "key": "release_triage",
        "setting": "a product team two days away from release with a bug list that looks longer than it really is",
        "role": "QA lead",
        "goal": "separate launch blockers from noisy but tolerable issues",
        "pain": "every bug feels urgent in isolation because each one has a different stakeholder",
        "artifact": "bug reports and test session notes",
        "tool": "an issue tracker and a release checklist",
        "metric": "open blocker count",
        "constraint": "the team can only land low-risk fixes before code freeze",
        "stakeholder": "the release manager",
        "task": "sort bug tickets by severity, reproducibility, and affected surface area",
        "fields": ("ticket_id", "severity", "repro_rate", "surface_area", "owner"),
        "notes": (
            "A bug in onboarding hurts more than a cosmetic issue in a rarely visited settings page.",
            "A reproducible medium bug can matter more than a vague high-severity report.",
            "The team needs a calmer story than everything is on fire.",
        ),
    },
    {
        "key": "crm_cleanup",
        "setting": "a sales team whose CRM is full of duplicate companies, stale deals, and half-finished notes",
        "role": "sales operations specialist",
        "goal": "make pipeline reporting believable again",
        "pain": "forecast calls get derailed because nobody agrees which accounts are real opportunities",
        "artifact": "CRM exports and call notes",
        "tool": "SQL and a cleanup spreadsheet",
        "metric": "active opportunities with clean ownership",
        "constraint": "sales reps ignore cleanup work that feels bureaucratic",
        "stakeholder": "the sales director",
        "task": "find duplicates, stale deals, and missing owners so cleanup starts with the highest-impact records",
        "fields": ("account_name", "owner", "stage", "last_activity", "amount"),
        "notes": (
            "Some duplicates exist because one rep entered a parent company and another entered a branch office.",
            "A forecast is only as useful as the records people trust enough to argue about.",
            "The cleanup plan has to save reps time or it will fail quietly.",
        ),
    },
    {
        "key": "moving_plan",
        "setting": "two roommates planning a move while both are still working full time",
        "role": "organized roommate",
        "goal": "avoid the final-week panic without turning the move into a second job",
        "pain": "small tasks like utility transfers slip because nobody owns them clearly",
        "artifact": "a shared checklist and moving quotes",
        "tool": "a notes app and a spreadsheet",
        "metric": "tasks completed before move day",
        "constraint": "both roommates only have evenings and one Saturday to prep",
        "stakeholder": "the roommate handling the landlord handoff",
        "task": "turn a rough moving checklist into dated tasks with owners and urgency flags",
        "fields": ("task", "owner", "due_date", "status", "urgent"),
        "notes": (
            "Packing is visible, but paperwork causes the ugly surprises.",
            "A calmer move comes from confirming earlier, not rushing harder at the end.",
            "People follow through better when the task list sounds realistic.",
        ),
    },
]

CATEGORIES = ("question_answering", "summarization", "code_generation", "reasoning", "transformation", "conversation")
DIFFICULTIES = ("easy", "medium", "hard")
VOICES = ("formal", "candid", "technical", "conversational", "managerial", "supportive")
SYSTEM_PROMPTS = {
    "question_answering": ("You explain ideas clearly and use the user's actual context.", "You answer directly, then add the practical detail that makes the answer useful."),
    "summarization": ("You summarize messy real-world information into something a busy person can act on.", "You keep the signal, remove the fluff, and preserve the caveats that matter."),
    "code_generation": ("You write clean, readable code and explain only what another developer needs.", "You favor straightforward, production-minded solutions over clever ones."),
    "reasoning": ("You reason carefully, make assumptions visible, and stay grounded in the details provided.", "You help people decide what to do next when a problem has several moving parts."),
    "transformation": ("You rewrite and restructure content without inventing facts.", "You turn rough writing into clearer, more useful outputs on demand."),
    "conversation": ("You sound human, thoughtful, and useful, especially when the user is stressed.", "You respond like a good teammate rather than a script."),
}


def pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def balanced_pairs(total: int, rng: random.Random) -> list[tuple[str, str]]:
    pairs = [(category, difficulty) for category in CATEGORIES for difficulty in DIFFICULTIES]
    out: list[tuple[str, str]] = []
    while len(out) < total:
        block = pairs[:]
        rng.shuffle(block)
        out.extend(block)
    return out[:total]


def clean_outputs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    for path in RAW_DIR.glob("*.md"):
        path.unlink()
    summary_path = SFT_DIR / "dataset_summary.json"
    if summary_path.exists():
        summary_path.unlink()


def make_raw_doc(index: int, scenario: dict, rng: random.Random) -> str:
    voice = VOICES[index % len(VOICES)]
    format_name = ("memo", "guide", "field_note", "faq", "postmortem", "journal", "case_study", "meeting_recap", "checklist")[index % 9]
    prefix = {
        "formal": "From an operational standpoint,",
        "candid": "If you spend even one week around this workflow,",
        "technical": "At the system level,",
        "conversational": "In plain language,",
        "managerial": "From a team perspective,",
        "supportive": "What helps people here is that",
    }[voice]
    lines = [
        f"# {scenario['goal'].split()[0].title()} and {scenario['key'].replace('_', ' ').title()} #{index + 1}",
        "",
        "## Context",
        "",
        f"{prefix} the {scenario['role']} is trying to {scenario['goal']}.",
        f"The friction is that {scenario['pain']}.",
        f"The team relies on {scenario['artifact']} and {scenario['tool']} because that is what fits the work as it exists today.",
        "",
    ]
    if format_name in {"guide", "checklist"}:
        lines.extend(
            [
                "## What usually matters most",
                "",
                f"- Keep {scenario['metric']} visible, but do not pretend it tells the whole story.",
                f"- Respect the fact that {scenario['constraint']}.",
                f"- Use {scenario['artifact']} as evidence, not as a substitute for judgment.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## What keeps showing up",
                "",
                f"{scenario['notes'][0]} {scenario['notes'][1]} {scenario['notes'][2]}",
                "",
            ]
        )
    if format_name in {"faq", "meeting_recap", "memo"}:
        lines.extend(
            [
                "## Practical takeaway",
                "",
                f"The next useful move is usually modest: clean one high-signal slice of {scenario['artifact']}, test whether it helps the {scenario['role']} act earlier, and explain the tradeoff to {scenario['stakeholder']}.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Closing note",
                "",
                f"The point is not to sound clever. The point is to give the team a calmer, more reliable way to improve {scenario['metric']} while staying honest about the messy reality of the work.",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_source(scenario: dict, difficulty: str) -> str:
    if difficulty == "easy":
        return (
            f"- Goal: {scenario['goal']}\n"
            f"- Problem: {scenario['pain']}\n"
            f"- Main evidence: {scenario['artifact']}\n"
            f"- Constraint: {scenario['constraint']}\n"
        )
    if difficulty == "medium":
        return (
            f"Team notes:\n"
            f"- {scenario['notes'][0]}\n"
            f"- {scenario['notes'][1]}\n"
            f"- {scenario['notes'][2]}\n"
            f"- Metric in focus: {scenario['metric']}\n"
            f"- Constraint: {scenario['constraint']}\n"
        )
    return (
        f"Leadership update draft: The team wants to {scenario['goal']}, but {scenario['pain']}. "
        f"Current evidence comes from {scenario['artifact']} and {scenario['tool']}. "
        f"We learned that {scenario['notes'][0]} We also learned that {scenario['notes'][1]} "
        f"and {scenario['notes'][2]} The visible number is {scenario['metric']}, but the working constraint is that {scenario['constraint']}."
    )


def build_pair(category: str, difficulty: str, scenario: dict) -> tuple[str, str]:
    fields = ", ".join(scenario["fields"])
    if category == "question_answering":
        if difficulty == "easy":
            return (
                f"I work in {scenario['setting']}. Can you explain what this work is really about in plain language and mention one mistake people make?",
                f"In your case, the work is about {scenario['goal']} while staying honest about the fact that {scenario['pain']}. "
                f"It is not just a theory problem; it is the daily discipline of using {scenario['artifact']} well enough to make calmer decisions. "
                f"A common mistake is treating {scenario['metric']} as the whole story and forgetting that {scenario['constraint']}.",
            )
        if difficulty == "medium":
            return (
                f"I am the {scenario['role']} in {scenario['setting']}. We want to {scenario['goal']}, but {scenario['pain']}. What should I look at first?",
                f"I would start with the part of {scenario['artifact']} that best exposes repeat patterns. "
                f"The first practical question is which cases are being handled in the wrong order and why. "
                f"Once that is visible, you can explain the tradeoff to {scenario['stakeholder']} in terms of {scenario['metric']} and the real-world limit that {scenario['constraint']}.",
            )
        return (
            f"Leadership wants one clean answer about this situation: {scenario['notes'][0]} {scenario['notes'][1]} {scenario['notes'][2]} How would you explain the problem, what should we test first, and what should we avoid oversimplifying?",
            f"I would explain this as a workflow problem with a measurement problem attached to it. The visible symptom sits in {scenario['metric']}, but the working issue is that {scenario['pain']}. "
            f"The best first test is to clean one high-signal slice of {scenario['artifact']} and see whether the {scenario['role']} can make better decisions sooner. "
            f"What I would not oversimplify is the fact that {scenario['constraint']}.",
        )

    if category == "summarization":
        source = build_source(scenario, difficulty)
        if difficulty == "easy":
            return (
                f"Summarize this note in 3 sentences for a busy teammate:\n\n{source}",
                f"The team wants to {scenario['goal']}, but {scenario['pain']}. Right now the signal is spread across {scenario['artifact']}, and any fix has to respect {scenario['constraint']}. If they handle this well, {scenario['metric']} should improve.",
            )
        if difficulty == "medium":
            return (
                f"I need a Slack-ready summary with one short overview and 3 action items:\n\n{source}",
                f"Overview: The team wants to {scenario['goal']}, but the workflow is brittle because {scenario['pain']}.\n\nAction items:\n- Clean the highest-signal slice of {scenario['artifact']}.\n- Check how {scenario['constraint']} changes the order of work.\n- Send {scenario['stakeholder']} a concise update tied to {scenario['metric']}.",
            )
        return (
            f"Turn this into an executive summary with Situation, Risks, and Recommended Next Step:\n\n{source}",
            f"Situation: The team is focused on {scenario['goal']}, but {scenario['pain']}. Evidence is split across {scenario['artifact']} and {scenario['tool']}.\n\nRisks: If the team chases only {scenario['metric']}, it may miss the effect of {scenario['constraint']}.\n\nRecommended Next Step: Clean one high-signal slice of the data, test an operational change, and brief {scenario['stakeholder']} in plain language.",
        )

    if category == "code_generation":
        if difficulty == "easy":
            return (
                f"Write a small Python function for {scenario['setting']}. Rows contain {fields}. I want to {scenario['task']}. Keep it readable.",
                "```python\nfrom collections import defaultdict\n\ndef summarize_rows(rows):\n    grouped = defaultdict(list)\n    for row in rows:\n        key = (row.get('" + scenario["fields"][1] + "'), row.get('" + scenario["fields"][-1] + "'))\n        grouped[key].append(row)\n    return sorted(({'group': key, 'count': len(items), 'rows': items} for key, items in grouped.items()), key=lambda item: item['count'], reverse=True)\n```\n\nThis stays intentionally plain so the logic is easy to audit.",
            )
        if difficulty == "medium":
            return (
                f"I need a Python script for {scenario['setting']}. Input is a CSV with columns {fields}. Please {scenario['task']}, handle missing values gracefully, and print a short summary.",
                "```python\nimport csv\nfrom collections import Counter, defaultdict\nfrom pathlib import Path\n\ndef clean(value):\n    return (value or '').strip()\n\ndef load_rows(path):\n    with Path(path).open('r', encoding='utf-8', newline='') as handle:\n        for row in csv.DictReader(handle):\n            yield {key: clean(value) for key, value in row.items()}\n\ndef summarize(path):\n    grouped = defaultdict(list)\n    counts = Counter()\n    for row in load_rows(path):\n        group_key = row.get('" + scenario["fields"][1] + "') or 'unknown'\n        grouped[group_key].append(row)\n        counts[group_key] += 1\n    for key, rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):\n        print(f'[{key}] rows={len(rows)} example={rows[0]}')\n    print('\\nTop groups:')\n    for key, count in counts.most_common(5):\n        print(f'- {key}: {count}')\n\nif __name__ == '__main__':\n    summarize('input.csv')\n```\n\nThis gives you a clean starting point before adding domain-specific ranking rules.",
            )
        return (
            f"I need production-minded Python for {scenario['setting']}. Read rows with {fields}, {scenario['task']}, and return a prioritized list we could review with {scenario['stakeholder']}. I need input validation and a visible scoring function.",
            "```python\nfrom dataclasses import dataclass\nfrom pathlib import Path\nimport csv\n\n@dataclass\nclass ReviewItem:\n    row: dict\n    score: float\n    reasons: list[str]\n\ndef clean(value):\n    return (value or '').strip()\n\ndef load_rows(path):\n    rows = []\n    with Path(path).open('r', encoding='utf-8', newline='') as handle:\n        for raw in csv.DictReader(handle):\n            row = {key: clean(value) for key, value in raw.items()}\n            if any(row.values()):\n                rows.append(row)\n    return rows\n\ndef score_row(row):\n    score = 0.0\n    reasons = []\n    if not row.get('" + scenario["fields"][0] + "'):\n        score += 1.5\n        reasons.append('missing_primary_id')\n    if not row.get('" + scenario["fields"][-1] + "'):\n        score += 1.0\n        reasons.append('missing_tail_field')\n    for key, value in row.items():\n        if value.lower() in {'urgent', 'high', 'critical'}:\n            score += 2.0\n            reasons.append(f'high_signal:{key}')\n    return ReviewItem(row=row, score=score, reasons=sorted(set(reasons)))\n\ndef prioritize(path):\n    return sorted((score_row(row) for row in load_rows(path)), key=lambda item: item.score, reverse=True)\n```\n\nThe scoring is simple on purpose so the review conversation stays understandable.",
        )

    if category == "reasoning":
        if difficulty == "easy":
            return (
                f"I need a practical opinion. We want to {scenario['goal']}, but {scenario['pain']}. Should I clean the data first or change the workflow first?",
                f"Start by cleaning the part of {scenario['artifact']} that directly affects decisions. If the team cannot see the pattern clearly, changing the workflow first usually just rearranges confusion. Once the signal is cleaner, you can decide whether the workflow still needs a deeper change.",
            )
        if difficulty == "medium":
            return (
                f"I need help prioritizing the next two weeks. We know that {scenario['notes'][0]} and {scenario['notes'][1]} The catch is {scenario['constraint']}. What would you do first?",
                f"I would do three things in order. First, build one trustworthy view of {scenario['metric']} from {scenario['artifact']}. Second, review the highest-friction cases and see whether they line up with the patterns in the notes. Third, turn that learning into a lightweight rule the {scenario['role']} can actually use under {scenario['constraint']}.",
            )
        return (
            f"I have to brief leadership. The visible symptom is {scenario['metric']}, but the team says the real issue is deeper: {scenario['notes'][0]} {scenario['notes'][1]} {scenario['notes'][2]} Given {scenario['constraint']}, what is the most defensible interpretation and what bad conclusion should we avoid?",
            f"The most defensible interpretation is that this is a workflow problem with a measurement problem attached to it. The visible symptom sits in {scenario['metric']}, but the operational issue is that {scenario['pain']}. The bad conclusion to avoid is assuming one policy change or one dashboard number will fix the situation by itself, especially because {scenario['constraint']}.",
        )

    if category == "transformation":
        rough = f"We need to improve this. Right now {scenario['pain']}. I think we should look at {scenario['artifact']}, but I also know {scenario['constraint']}. {scenario['stakeholder']} wants an update soon."
        if difficulty == "easy":
            return (
                f"Rewrite this so it sounds clearer and more professional without becoming stiff:\n\n{rough}",
                f"We need to tighten our approach here. The current issue is that {scenario['pain']}. My recommendation is to review the signal inside {scenario['artifact']} first, while keeping in mind that {scenario['constraint']}. I also want to send {scenario['stakeholder']} a concise update once we have a concrete next step.",
            )
        if difficulty == "medium":
            return (
                f"Turn this into a manager-ready update with Problem, Why It Matters, and Next Step:\n\n{rough}",
                f"Problem: {scenario['pain']}.\n\nWhy It Matters: The issue is already affecting {scenario['metric']} and making life harder for the {scenario['role']}.\n\nNext Step: Review the strongest signals inside {scenario['artifact']}, test one practical adjustment, and share a short update with {scenario['stakeholder']}.",
            )
        return (
            f"Transform this into a compact operating brief with Situation, Constraints, Recommendation, and Open Questions. Do not invent facts:\n\n{rough}\n\nExtra context: {scenario['notes'][0]} {scenario['notes'][1]}",
            f"Situation: The team wants to {scenario['goal']}, but {scenario['pain']}.\n\nConstraints: {scenario['constraint']}. We also know that {scenario['notes'][0].lower()} and {scenario['notes'][1].lower()}\n\nRecommendation: Clean one high-signal slice of {scenario['artifact']}, test an operational change, and frame the update around {scenario['metric']} plus observed workflow impact.\n\nOpen Questions: Which signals are trustworthy enough to guide action right now, and what level of detail does {scenario['stakeholder']} actually need?",
        )

    if difficulty == "easy":
        return (
            f"I am feeling a bit stuck. I help with {scenario['setting']}, and {scenario['pain']}. Can you help me think about the next step without making this more complicated?",
            f"Absolutely. Keep the next step small and concrete. Start by finding the part of {scenario['artifact']} that best reflects the problem, then ask whether it helps you {scenario['goal']} or mostly creates noise. You do not need the perfect system today; you need one clearer view of the work.",
        )
    if difficulty == "medium":
        return (
            f"I need to send a thoughtful reply to my manager. We are dealing with this: {scenario['notes'][0]} and {scenario['stakeholder']} wants a quick answer. What should I say if I want to sound calm, honest, and useful?",
            f"You could say: \"I have reviewed the situation, and it looks less like a one-off mistake and more like a repeat pattern in how the work is flowing. The fastest useful next step is to review the signal inside {scenario['artifact']}, test one practical adjustment, and measure the effect on {scenario['metric']}. I want to move quickly without oversimplifying the fact that {scenario['constraint']}.\"",
        )
    return (
        f"I am preparing for a tense meeting. Leadership wants certainty, but the truth is more complicated: {scenario['notes'][0]} {scenario['notes'][1]} {scenario['notes'][2]} Can you help me frame this in a way that is human, credible, and not defensive?",
        f"Start by naming the shared goal: the team is trying to {scenario['goal']}. Then describe the working reality: {scenario['pain']}. Acknowledge the pressure without pretending certainty you do not have. Close with a practical next step tied to {scenario['artifact']} and {scenario['metric']}. That combination usually sounds responsible and human.",
    )


def make_record(category: str, difficulty: str, scenario: dict, record_id: str, rng: random.Random) -> dict:
    user, assistant = build_pair(category, difficulty, scenario)
    return {
        "id": record_id,
        "category": category,
        "difficulty": difficulty,
        "scenario": scenario["key"],
        "system": pick(rng, SYSTEM_PROMPTS[category]),
        "user": user,
        "assistant": assistant,
    }


def generate_split(
    prefix: str,
    total: int,
    seed: int,
    offset: int,
    id_offset: int = 0,
    pbar: tqdm | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    pairs = balanced_pairs(total, rng)
    rows: list[dict] = []
    for index, (category, difficulty) in enumerate(pairs):
        scenario = SCENARIOS[(index + offset) % len(SCENARIOS)]
        local_rng = random.Random(seed + index * 9973)
        row = make_record(category, difficulty, scenario, f"{prefix}_{id_offset + index + 1:04d}", local_rng)
        rows.append(row)
        if pbar is not None:
            pbar.update(1)
    return rows


def write_summary(raw_count: int, train_rows: list[dict], val_rows: list[dict]) -> None:
    def summarize(rows: list[dict]) -> dict:
        return {
            "records": len(rows),
            "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
            "difficulties": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
            "category_difficulty": dict(sorted(Counter(f"{row['category']}::{row['difficulty']}" for row in rows).items())),
        }

    summary = {"raw_markdown_documents": raw_count, "train": summarize(train_rows), "val": summarize(val_rows)}
    (SFT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


async def _write_raw_doc(path: Path, content: str) -> None:
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)


async def _write_jsonl(path: Path, rows: list[dict], pbar: tqdm) -> None:
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        for row in rows:
            await f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pbar.update(1)


async def _main(args: argparse.Namespace) -> None:
    clean_outputs()

    # --- Step 1: write raw Markdown docs concurrently ---
    doc_jobs = []
    for index in range(args.raw_count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        doc_rng = random.Random(args.seed * 101 + index)
        content = make_raw_doc(index, scenario, doc_rng)
        path = RAW_DIR / f"{index + 1:03d}_{scenario['key']}.md"
        doc_jobs.append(_write_raw_doc(path, content))

    with tqdm(total=args.raw_count, desc="raw docs  ", unit="doc", ncols=72) as raw_pbar:
        for coro in asyncio.as_completed(doc_jobs):
            await coro
            raw_pbar.update(1)

    # --- Step 2: generate train + val splits concurrently in threads ---
    train_pbar = tqdm(total=args.train_count, desc="train rows", unit="row", position=0, ncols=72)
    val_pbar   = tqdm(total=args.val_count,   desc="val rows  ", unit="row", position=1, ncols=72)
    try:
        train_rows, val_rows = await asyncio.gather(
            asyncio.to_thread(
                generate_split, "train", args.train_count,
                args.seed * 17 + 3, 0, 0, train_pbar,
            ),
            asyncio.to_thread(
                generate_split, "train", args.val_count,
                args.seed * 23 + 5, 5, args.train_count, val_pbar,
            ),
        )
    finally:
        train_pbar.close()
        val_pbar.close()

    # --- Step 3: write JSONL files concurrently ---
    train_write_pbar = tqdm(total=len(train_rows), desc="write train", unit="row", position=0, ncols=72)
    val_write_pbar   = tqdm(total=len(val_rows),   desc="write val  ", unit="row", position=1, ncols=72)
    try:
        await asyncio.gather(
            _write_jsonl(SFT_DIR / "sample_train.jsonl", train_rows, train_write_pbar),
            _write_jsonl(SFT_DIR / "sample_val.jsonl",   val_rows,   val_write_pbar),
        )
    finally:
        train_write_pbar.close()
        val_write_pbar.close()

    # --- Step 4: write summary ---
    write_summary(args.raw_count, train_rows, val_rows)

    print(f"raw_docs={args.raw_count}")
    print(f"train_rows={len(train_rows)}")
    print(f"val_rows={len(val_rows)}")
    print(f"raw_dir={RAW_DIR}")
    print(f"sft_train={SFT_DIR / 'sample_train.jsonl'}")
    print(f"sft_val={SFT_DIR / 'sample_val.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overwrite data/raw and data/sft with more natural synthetic datasets.")
    parser.add_argument("--raw-count",   type=int, default=180)
    parser.add_argument("--train-count", type=int, default=1080)
    parser.add_argument("--val-count",   type=int, default=180)
    parser.add_argument("--seed",        type=int, default=11)
    return parser.parse_args()


def main() -> None:
    asyncio.run(_main(parse_args()))


if __name__ == "__main__":
    main()

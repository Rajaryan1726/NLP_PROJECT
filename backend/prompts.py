"""All prompt templates in one place."""
import json

# Target summary size = a share of the input length, capped, so a summary is never
# longer than a short input (that would give a negative compression ratio).
LENGTH_RATIO = {"short": 0.2, "medium": 0.35, "long": 0.5}
LENGTH_CAP = {"short": 60, "medium": 120, "long": 220}
BULLETS = {"short": 3, "medium": 5, "long": 8}
STYLE_GUIDE = {
    "paragraph": "a single paragraph",
    "bullets": "bullet points, each line starting with '- '",
}

HINGLISH_RULE = (
    "Write in Hinglish: Hindi written in Roman (Latin) script in a natural code-mixed style, "
    "e.g. 'Sarkar ne nayi scheme launch ki hai'. Never use Devanagari script. Never write pure English."
)


def _length(length: str, style: str, text: str) -> str:
    words = max(15, min(LENGTH_CAP[length], round(len(text.split()) * LENGTH_RATIO[length])))
    if style == "bullets":
        bullets = max(2, min(BULLETS[length], words // 10))
        return f"{bullets} bullet points, at most {words} words in total"
    return f"at most {words} words"


# ---------------------------------------------------------------- summarization

def standard_prompt(text: str, length: str, style: str) -> list[dict]:
    """The baseline: a plain summarization instruction."""
    return [{
        "role": "user",
        "content": f"Summarize this text in Hinglish (Roman script), {_length(length, style, text)}, "
                   f"as {STYLE_GUIDE[style]}.\n\nText:\n{text}",
    }]


FACTUALITY_SYSTEM = f"""You are a careful news summarizer for Hindi, English and Hindi-English code-mixed text.

Rules:
1. First list the key facts from the source: entities (people, organisations, places, schemes), numbers, dates, events and who-did-what.
2. Then write the summary using ONLY those facts. Respect the word limit: if not every fact fits, keep the most important ones.
3. Copy every number, date and amount exactly as in the source (e.g. keep "4,500 crore", do not round or convert). Keep every name, but write it in Roman script (transliterate: "विद्या सेतु" -> "Vidya Setu", "करोड़" -> "crore", "अगस्त" -> "August").
4. Never add information that is not in the source: no guesses, no background knowledge, no opinions.
5. Keep negations and relationships exactly (who did what to whom).
6. The source may mix Hindi, English and romanized Hindi. Read code-mixed phrases carefully; do not mistranslate them.
7. {HINGLISH_RULE}

Output format (exactly):
FACTS:
- <fact 1>
- <fact 2>
...
SUMMARY:
<the summary, 100% Roman script, no Devanagari characters at all>"""


def factuality_prompt(text: str, length: str, style: str, memory_rules: list[str]) -> list[dict]:
    user = f"Summarize the source as {STYLE_GUIDE[style]}, length: {_length(length, style, text)}.\n"
    if memory_rules:
        user += "\nPast corrections from this user (avoid repeating these mistakes):\n"
        user += "\n".join(f"- {r}" for r in memory_rules) + "\n"
    user += f"\nSource:\n{text}"
    return [{"role": "system", "content": FACTUALITY_SYSTEM}, {"role": "user", "content": user}]


def corrective_prompt(text: str, length: str, style: str, previous_summary: str,
                      problems: list[str], memory_rules: list[str]) -> list[dict]:
    """Regenerate after verification failed. `problems` explain each failed claim with its evidence."""
    user = (
        f"Your previous summary had factual errors.\n\nPrevious summary:\n{previous_summary}\n\n"
        "Problems found by the fact checker:\n" + "\n".join(f"- {p}" for p in problems) + "\n\n"
        "Rewrite the summary so that every statement is directly supported by the source. "
        "Fix or remove the wrong statements; keep the correct ones.\n"
        f"Format: {STYLE_GUIDE[style]}, length: {_length(length, style, text)}.\n"
    )
    if memory_rules:
        user += "\nPast corrections from this user:\n" + "\n".join(f"- {r}" for r in memory_rules) + "\n"
    user += f"\nSource:\n{text}"
    return [{"role": "system", "content": FACTUALITY_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- verification helpers

def claims_prompt(summary: str) -> list[dict]:
    return [{
        "role": "system",
        "content": (
            "Split the summary into atomic factual claims. Each claim states exactly one fact, is short, "
            "and is self-contained (replace pronouns with the names they refer to; check carefully who "
            "'unhone' / 'they' / 'he' is in the summary, e.g. the hackers vs the victim). Keep numbers, dates and "
            "names exactly. Keep each claim in the summary's language (Hinglish), and also give a faithful "
            "English translation that keeps numbers and units unchanged (keep 'crore', 'lakh').\n"
            'Return JSON: {"claims": [{"claim": "...", "claim_en": "..."}]}'
        ),
    }, {"role": "user", "content": summary}]


def translate_prompt(texts: list[str]) -> list[dict]:
    return [{
        "role": "system",
        "content": (
            "Translate each input text (Hindi, romanized Hindi or code-mixed) into faithful English. "
            "Do not summarize, add or drop anything. Keep numbers, units ('crore', 'lakh'), dates and names "
            "exactly. If a text is already English, return it unchanged.\n"
            f'Return JSON: {{"translations": [...]}} with exactly {len(texts)} strings in the same order.'
        ),
    }, {"role": "user", "content": json.dumps({"texts": texts}, ensure_ascii=False)}]


def entities_prompt(summary: str) -> list[dict]:
    return [{
        "role": "system",
        "content": (
            "List the proper names in the text: people, organisations, places, schemes, products and named "
            "events, written exactly as in the text. Do NOT list common nouns (e.g. 'sarkar', 'students', "
            "'minister', 'police'), numbers, amounts or dates.\n"
            'Return JSON: {"entities": [{"text": "...", "type": "PERSON|ORG|LOC|EVENT|OTHER"}]}'
        ),
    }, {"role": "user", "content": summary}]


def memory_rule(claim_text: str, note: str) -> str:
    """How a user correction is phrased when stored in Mem0."""
    note = note.strip() or "the user marked it as wrong"
    return f"Previously a summary claim \"{claim_text}\" was marked wrong because: {note}"

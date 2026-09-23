"""Text cleaning, language detection, chunking and number/date normalization."""
import re
import unicodedata

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Frequent Hindi function words written in Roman script (words that are also English,
# like "the", "me", "log", "kar", are left out). If enough of them appear in
# Latin-script text, it is romanized Hindi / Hinglish rather than English.
ROMAN_HINDI_WORDS = {
    "hai", "hain", "tha", "thi", "ki", "ka", "ke", "ko", "ne", "se", "mein",
    "aur", "ya", "par", "pe", "bhi", "nahi", "nahin", "kya", "kyun", "ye", "yeh", "woh", "vo",
    "jo", "jab", "tab", "kiya", "kiye", "gaya", "gayi", "gaye", "raha", "rahi", "rahe", "hoga",
    "hogi", "karna", "karne", "liye", "wala", "wale", "wali", "abhi", "sabhi", "unke",
    "unka", "iske", "uske", "apne", "diya", "diye", "saal", "logon", "jaisa", "sirf",
}


def clean_text(text: str) -> str:
    """NFC-normalize, convert Devanagari digits to ASCII, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(DEVANAGARI_DIGITS)
    text = text.replace("​", "").replace("‌", "").replace("‍", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return 'hindi', 'english' or 'code_mixed' from the script ratio + romanized-Hindi words."""
    dev = len(DEVANAGARI_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    if dev + lat == 0:
        return "english"
    dev_ratio = dev / (dev + lat)
    if dev_ratio >= 0.85:
        return "hindi"
    if dev_ratio >= 0.15:
        return "code_mixed"  # real mix of both scripts
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "english"
    hindi_hits = sum(1 for w in words if w in ROMAN_HINDI_WORDS)
    return "code_mixed" if hindi_hits / len(words) >= 0.08 else "english"


def word_count(text: str) -> int:
    return len(text.split())


# A sentence is a run of text up to । ? ! . or a newline. A '.' followed by a digit
# is a decimal point ("4.5"), not a sentence end.
SENTENCE_RE = re.compile(r"(?:[^।?!.\n]|\.(?=\d))+(?:[।?!.]+|\n|$)")


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return (sentence, start, end) with character offsets into `text`."""
    sentences = []
    for m in SENTENCE_RE.finditer(text):
        piece = m.group(0)
        if not piece.strip():
            continue
        start = m.start() + (len(piece) - len(piece.lstrip()))
        end = m.start() + len(piece.rstrip())
        sentences.append((text[start:end], start, end))
    return sentences


def chunk_text(text: str, max_sentences: int = 3, max_words: int = 70) -> list[dict]:
    """Group sentences into chunks of 2-3 sentences (fewer if the sentences are long)."""
    chunks, current = [], []
    for sent in split_sentences(text):
        current.append(sent)
        words = sum(word_count(s[0]) for s in current)
        if len(current) >= max_sentences or (len(current) >= 2 and words >= max_words) or words >= max_words * 1.5:
            chunks.append(current)
            current = []
    if current:
        # avoid a lone trailing sentence: 3+1 becomes 2+2, 2+1 becomes 3
        if chunks and len(current) == 1:
            if len(chunks[-1]) >= 3:
                current.insert(0, chunks[-1].pop())
                chunks.append(current)
            else:
                chunks[-1].extend(current)
        else:
            chunks.append(current)
    result = []
    for i, group in enumerate(chunks):
        start, end = group[0][1], group[-1][2]
        result.append({"chunk_id": i, "text": text[start:end], "start": start, "end": end})
    return result


# ---------------------------------------------------------------- numbers

MULTIPLIERS = {
    "crore": 1e7, "crores": 1e7, "cr": 1e7, "करोड़": 1e7, "करोड": 1e7, "karod": 1e7, "karor": 1e7,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "लाख": 1e5, "laakh": 1e5,
    "arab": 1e9, "अरब": 1e9, "kharab": 1e11, "खरब": 1e11,
    "hazar": 1e3, "hazaar": 1e3, "hajar": 1e3, "हजार": 1e3, "हज़ार": 1e3, "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "मिलियन": 1e6,
    "billion": 1e9, "bn": 1e9, "बिलियन": 1e9,
    "trillion": 1e12,
}
_MULT_ALT = "|".join(sorted((re.escape(k) for k in MULTIPLIERS), key=len, reverse=True))
NUMBER_RE = re.compile(rf"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*({_MULT_ALT})(?![\wऀ-ॿ]))?", re.IGNORECASE)

# Small number words only count on the SOURCE side (e.g. "दो साल" vs summary "2 saal").
NUMBER_WORDS = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "सौ": 100,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "hundred": 100,
    "ek": 1, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhah": 6, "saat": 7,
    "aath": 8, "nau": 9, "das": 10,
}


def parse_numbers(text: str) -> list[tuple[str, float]]:
    """Find numbers like '4,500 crore', '2 करोड़', '3.5 million' -> (raw text, value)."""
    out = []
    for m in NUMBER_RE.finditer(text):
        value = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit:
            value *= MULTIPLIERS[unit]
        out.append((m.group(0).strip(), value))
    return out


def bare_numbers(text: str) -> list[float]:
    """The digits alone, ignoring any unit word."""
    return [float(m.group(1).replace(",", "")) for m in NUMBER_RE.finditer(text)]


def number_words(text: str) -> set[float]:
    tokens = re.findall(r"[\wऀ-ॿ]+", text.lower())
    return {float(NUMBER_WORDS[t]) for t in tokens if t in NUMBER_WORDS}


# ---------------------------------------------------------------- dates

MONTHS = {
    1: ["january", "jan", "जनवरी", "janvari", "janwari"],
    2: ["february", "feb", "फरवरी", "फ़रवरी", "farvari", "farwari"],
    3: ["march", "mar", "मार्च", "march"],
    4: ["april", "apr", "अप्रैल", "aprail"],
    5: ["may", "मई"],
    6: ["june", "jun", "जून"],
    7: ["july", "jul", "जुलाई", "julai"],
    8: ["august", "aug", "अगस्त", "agast"],
    9: ["september", "sep", "sept", "सितंबर", "सितम्बर", "sitambar"],
    10: ["october", "oct", "अक्टूबर", "aktubar"],
    11: ["november", "nov", "नवंबर", "नवम्बर", "navambar"],
    12: ["december", "dec", "दिसंबर", "दिसम्बर", "disambar"],
}
MONTH_LOOKUP = {name: num for num, names in MONTHS.items() for name in names}
_MONTH_ALT = "|".join(sorted((re.escape(k) for k in MONTH_LOOKUP), key=len, reverse=True))
_B = r"(?![\wऀ-ॿ])"  # word boundary that also works for Devanagari

DATE_PATTERNS = [
    # 2023-08-15
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    # 15/08/2023 or 15-08-2023 or 15.08.2023 (Indian day-first order)
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
    # 15 August 2023 / 15 अगस्त / 15th Aug, 2023
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT}){_B},?(?:\s+(\d{{4}}))?", re.IGNORECASE),
     lambda m: (int(m[3]) if m[3] else None, MONTH_LOOKUP[m[2].lower()], int(m[1]))),
    # August 15, 2023
    (re.compile(rf"(?<![\wऀ-ॿ])({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?(?:\s+(\d{{4}}))?", re.IGNORECASE),
     lambda m: (int(m[3]) if m[3] else None, MONTH_LOOKUP[m[1].lower()], int(m[2]))),
    # August 2023 / अगस्त 2023
    (re.compile(rf"(?<![\wऀ-ॿ])({_MONTH_ALT})\s+(\d{{4}})\b", re.IGNORECASE),
     lambda m: (int(m[2]), MONTH_LOOKUP[m[1].lower()], None)),
]


def parse_dates(text: str) -> tuple[list[tuple[str, tuple]], str]:
    """Return ([(raw, (year, month, day))], text with the dates blanked out).

    Missing parts are None. Dates are blanked so their digits are not counted again as numbers.
    """
    found = []
    for pattern, build in DATE_PATTERNS:
        for m in pattern.finditer(text):
            try:
                y, mo, d = build(m)
            except (KeyError, ValueError):
                continue
            if (mo and not 1 <= mo <= 12) or (d and not 1 <= d <= 31):
                continue
            found.append((m.group(0), (y, mo, d)))
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return found, text


def dates_match(summary_date: tuple, source_date: tuple) -> bool:
    """Every part present in the summary date must equal the source date's part (if it has one)."""
    for s, t in zip(summary_date, source_date):
        if s is not None and t is not None and s != t:
            return False
    # the source must at least share the month or the year
    return any(s is not None and s == t for s, t in zip(summary_date, source_date))


def numbers_match(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0)

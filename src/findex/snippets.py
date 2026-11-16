import json
import re
from pathlib import Path

from .tokenize import tokenize


def make_snippet(
    document_path: str | Path,
    query_terms: list[str],
    context_chars: int = 80,
    *,
    line_offset: int | None = None,
    phrase_terms: list[str] | None = None,
) -> str:
    """Return a short relevant excerpt with phrase or term matches highlighted."""
    path = Path(document_path)

    try:
        if line_offset is None:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            with path.open("rb") as source:
                source.seek(line_offset)
                record = json.loads(source.readline().decode("utf-8"))
            text = str(record.get("text", ""))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return "[Document text unavailable]"

    terms = list(
        dict.fromkeys(
            term for query_term in query_terms for term in tokenize(query_term) if term
        )
    )
    phrase = [term for term in (phrase_terms or []) if term]

    if not text or not terms:
        return text[: 2 * context_chars]

    phrase_pattern = None
    if len(phrase) > 1:
        # A phrase is a sequence of tokens; punctuation/markup between words
        # is allowed (for example, LaTeX commands separated by backslashes).
        phrase_pattern = re.compile(
            r"(?<!\w)" + r"[\W_]+".join(re.escape(term) for term in phrase) + r"(?!\w)",
            re.IGNORECASE,
        )

    phrase_match = phrase_pattern.search(text) if phrase_pattern else None
    if phrase_match:
        match_start, match_end = phrase_match.span()
        start = max(0, match_start - context_chars)
        end = min(len(text), match_end + context_chars)
        snippet = text[start:end]
        if start:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        # Highlight the entire phrase as one match, including separators.
        return phrase_pattern.sub(
            lambda found: f"**{found.group(0)}**", snippet, count=1
        )

    alternatives = "|".join(
        re.escape(term) for term in sorted(set(terms), key=len, reverse=True)
    )
    pattern = re.compile(rf"(?<!\w)({alternatives})(?!\w)", re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return text[: 2 * context_chars]

    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)
    snippet = text[start:end]
    if start:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."
    return pattern.sub(lambda found: f"**{found.group(0)}**", snippet)

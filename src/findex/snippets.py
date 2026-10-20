
import re
from pathlib import Path

from .tokenize import tokenize


def make_snippet(
    document_path: str | Path,
    query_terms: list[str],
    context_chars: int = 80,
) -> str:
    path = Path(document_path)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "[Document text unavailable]"

    terms = {
        term
        for query_term in query_terms
        for term in tokenize(query_term)
        if term
    }

    if not text or not terms:
        return text[: 2 * context_chars]

    alternatives = "|".join(
        re.escape(term)
        for term in sorted(terms, key=len, reverse=True)
    )

    # Unicode-aware boundaries: do not match a term inside a larger word.
    pattern = re.compile(
        rf"(?<!\w)({alternatives})(?!\w)",
        re.IGNORECASE,
    )

    match = pattern.search(text)

    if match is None:
        return text[: 2 * context_chars]

    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)

    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet

    if end < len(text):
        snippet += "..."

    return pattern.sub(
        lambda found: f"**{found.group(0)}**",
        snippet,
    )

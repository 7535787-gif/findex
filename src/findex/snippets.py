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

    terms = list(
        dict.fromkeys(
            term for query_term in query_terms for term in tokenize(query_term) if term
        )
    )
    phrase = [term for term in (phrase_terms or []) if term]

    if not terms:
        return ""

    phrase_pattern = None
    if len(phrase) > 1:
        phrase_pattern = re.compile(
            r"(?<!\w)" + r"[\W_]+".join(re.escape(term) for term in phrase) + r"(?!\w)",
            re.IGNORECASE,
        )

    alternatives = "|".join(
        re.escape(term) for term in sorted(set(terms), key=len, reverse=True)
    )
    pattern = re.compile(rf"(?<!\w)({alternatives})(?!\w)", re.IGNORECASE)

    try:
        with path.open("rb") as source:
            if line_offset is not None:
                source.seek(line_offset)
                record = json.loads(source.readline().decode("utf-8"))
                text = str(record.get("text", ""))
                return _extract_snippet(text, pattern, phrase_pattern, context_chars)

            overlap = max(
                context_chars * 2,
                max((len(term) for term in phrase), default=0) + context_chars * 2,
                4096,
            )
            carry = ""
            position = 0

            while True:
                chunk = source.read(64 * 1024)
                if not chunk:
                    break

                text = carry + chunk.decode("utf-8", errors="replace")
                search_from = max(0, len(carry) - overlap)

                phrase_match = (
                    phrase_pattern.search(text, search_from)
                    if phrase_pattern is not None
                    else None
                )
                match = phrase_match or pattern.search(text, search_from)

                if match is not None:
                    start = max(0, match.start() - context_chars)
                    end = min(len(text), match.end() + context_chars)
                    snippet = text[start:end]

                    if start > 0 or position > 0:
                        snippet = "..." + snippet
                    if end < len(text) or len(chunk) == 64 * 1024:
                        snippet += "..."

                    if phrase_match is not None and phrase_pattern is not None:
                        return phrase_pattern.sub(
                            lambda found: f"**{found.group(0)}**",
                            snippet,
                            count=1,
                        )

                    return pattern.sub(
                        lambda found: f"**{found.group(0)}**",
                        snippet,
                    )

                carry = text[-overlap:]
                position += len(chunk)

    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return "[Document text unavailable]"

    return "[No matching excerpt found]"



def _extract_snippet(
    text: str,
    pattern: re.Pattern[str],
    phrase_pattern: re.Pattern[str] | None,
    context_chars: int,
) -> str:
    """Extract and highlight a matching excerpt from an in-memory text."""
    match = None
    active_pattern = pattern

    if phrase_pattern is not None:
        match = phrase_pattern.search(text)
        if match is not None:
            active_pattern = phrase_pattern

    if match is None:
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

    return active_pattern.sub(
        lambda found: f"**{found.group(0)}**",
        snippet,
        count=1,
    )
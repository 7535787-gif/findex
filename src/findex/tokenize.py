import re
import unicodedata
from typing import Iterator


WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’ʼ][^\W\d_]+)*")


def tokenize(text: str) -> Iterator[str]:
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()

    for match in WORD_PATTERN.finditer(text):
        yield match.group()
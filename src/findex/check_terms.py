"""Small diagnostic helper for inspecting corpus terms."""

import logging
from pathlib import Path

from .index_model import open_index
from .tokenize import tokenize

logger = logging.getLogger(__name__)


def main() -> None:
    with open_index(Path("index.bin")) as index:
        logger.info("Sample terms: %s", list(index)[:40])
        query_terms = list(tokenize("кібербезпека та захист інформації"))
        logger.info("Query tokens: %s", query_terms)
        logger.info(
            "Matching terms: %s", [term for term in query_terms if term in index]
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

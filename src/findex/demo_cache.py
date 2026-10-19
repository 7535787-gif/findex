
from pathlib import Path

from findex.cached_search import demo_cache
from findex.index_model import open_index

with open_index(Path("index.bin")) as index:
    demo_cache(index, "python")

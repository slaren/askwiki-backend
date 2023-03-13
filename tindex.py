import bisect
import logging
from typing import List

import faiss
import numpy as np

import model
from util import perf_logger

logger = logging.getLogger(__name__)


logger.info("Loading index")
index = faiss.read_index("index.faiss")

logger.info("Loading titles")
with open("titles.txt", "r") as f:
    titles = f.read().splitlines()

logger.info("Loading complete")

# TODO: sort titles in file and rebuild faiss index
titles_s = sorted(titles, key=lambda x: x.lower())


def suggest_titles(query: str, n: int = 10) -> List[str]:
    with perf_logger("Computing query embedding", logger):
        query_embedding = model.encode([query])

    # The index is not completely accurate, so to improve the results we search
    # for more candidates and select the best
    # The number of candidates is a rounded to a multiple of the batch size to
    # improve efficiency
    batch_size = 64
    _, I = index.search(query_embedding, int(n * 10 / batch_size + 0.5) * batch_size)

    candidates = [titles[i] for i in I[0]]

    # Embeddings could be precomputed and stored in a database, but may require too
    # much storage for a cheap server. But computing them without a GPU may be too slow
    # too. Find better alternatives.
    with perf_logger("Computing candidate embeddings", logger):
        cand_embeddings = model.encode(candidates, batch_size=batch_size)

    # Calculate similarity between the query and all candidates
    similarity = np.dot(query_embedding, cand_embeddings.T) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(cand_embeddings, axis=1)
    )

    # Sort the candidates by similarity and return the top n
    closest_titles = [titles[I[0][i]] for i in np.argsort(similarity[0])[::-1][:n]]

    return closest_titles


def search_titles(query: str, n: int = 10) -> List[str]:
    with perf_logger("Searching titles", logger):
        # Perform a binary search to find the first title that starts with the query
        i = bisect.bisect_left(titles_s, query.lower(), key=lambda s: s.lower())
        results = titles_s[i : i + n]

    return results

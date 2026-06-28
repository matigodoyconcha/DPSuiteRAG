"""Movielens-1M dataset loader, formatted for RAG benchmarking.

Source: Harper & Konstan, "The MovieLens Datasets: History and Context",
ACM TiiS 2015. HuggingFace mirror: ``ashraq/movielens_ratings`` or local
``ml-1m.zip`` from https://files.grouplens.org/datasets/movielens/ml-1m.zip.

The DP-SynRAG paper (arXiv:2510.06719v2) uses Movielens-1M for a "user-profile
to movie title" benchmark. We replicate that protocol:

    documents: one document per user, listing the movies that user rated
        highly (rating >= 4). Each document is a small user profile, which
        is the privacy unit (one row = one user).
    queries:   one query per held-out user, asking "Suggest a movie this
        user might like" with the highest-rated unseen movie as the gold
        answer.

The loader tries HuggingFace first and falls back to downloading the raw zip
when no compatible HF dataset is available.
"""

from __future__ import annotations

import io
import os
import random
import urllib.request
import zipfile
from collections import defaultdict
from typing import Dict, List

from bench_datasets.base_dataset import BaseDataset


_GROUPLENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


class MovielensDataset(BaseDataset):
    """Movielens-1M user-profile RAG benchmark."""

    # Task: movie recommendation from a user profile.
    TASK_INSTRUCTION = (
        "Based on the user's profile in the context, recommend a movie. Give the "
        "movie title only"
    )

    def __init__(
        self,
        index_split: str = "train",
        test_split: str = "validation",
        max_index_docs: int | None = 500,
        max_test_queries: int | None = 100,
        random_seed: int = 42,
        cache_dir: str | None = None,
        min_high_ratings: int = 5,
        top_unseen_pool: int = 30,
        top_k_favorites: int = 10,
        seed_size: int = 3,
    ):
        self.index_split = index_split
        self.test_split = test_split
        self.max_index_docs = max_index_docs
        self.max_test_queries = max_test_queries
        self.random_seed = random_seed
        self.cache_dir = cache_dir or os.path.expanduser(
            "~/.cache/dp_benchmark_rag/movielens-1m"
        )
        self.min_high_ratings = int(min_high_ratings)
        self.top_unseen_pool = int(top_unseen_pool)
        # DPSYN-09: the gold is the user's top-k favorites; the query shows a small
        # seed of them and the recommendation is scored (hit_rate) against the
        # held-out remainder.
        self.top_k_favorites = int(top_k_favorites)
        self.seed_size = int(seed_size)

        self._documents: list[str] = []
        self._queries: list[dict] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        movies, users_ratings, movie_popularity = self._fetch_raw()
        rng = random.Random(self.random_seed)

        user_ids = [uid for uid, ratings in users_ratings.items() if len(ratings) >= self.min_high_ratings]
        rng.shuffle(user_ids)

        def _profile_doc(uid: int) -> str | None:
            high = [mid for mid, rating in users_ratings[uid].items() if rating >= 4]
            if len(high) < self.min_high_ratings:
                return None
            title_list = [movies[mid] for mid in high if mid in movies]
            if not title_list:
                return None
            return (
                f"User {uid} has highly rated the following movies: "
                + "; ".join(title_list)
                + "."
            )

        def _query(uid: int) -> dict | None:
            # DPSYN-09: gold = the held-out user's TOP-k favorites (ranked by rating,
            # then popularity). The query shows only a small seed of them for
            # retrieval; the recommendation is scored (hit_rate) against the remaining
            # held-out favorites, so a trivial echo of the seed cannot score.
            ratings = users_ratings[uid]
            high = [mid for mid, rating in ratings.items() if rating >= 4]
            if len(high) < self.min_high_ratings:
                return None
            ranked = sorted(
                (mid for mid in high if mid in movies),
                key=lambda m: (ratings[m], movie_popularity.get(m, 0)),
                reverse=True,
            )
            top_favorites = [movies[mid] for mid in ranked[: self.top_k_favorites]]
            seed = top_favorites[: self.seed_size]
            held_out = top_favorites[self.seed_size :]
            if not seed or not held_out:
                return None  # need favorites beyond the shown seed to score against
            profile_brief = "; ".join(seed)
            return {
                "query": (
                    "Given a user who liked these movies: "
                    f"{profile_brief}. Suggest one movie they would also enjoy."
                ),
                "answer": held_out[0],   # single gold (compat/debug)
                "favorites": held_out,   # gold SET for the hit_rate metric
            }

        # Reserve a DISJOINT set of users for queries first, so an oversized
        # max_index_docs (e.g. larger than the whole user base) can never starve the
        # query set. The previous logic only emitted queries once the document budget
        # was full, so max_index_docs >= the ~6040 ML-1M users yielded zero queries
        # and aborted the run. Queries are carved from the front of the shuffled
        # users; documents are filled from the remaining users.
        queries: list[dict] = []
        used_for_queries: set[int] = set()
        for uid in user_ids:
            if self.max_test_queries is not None and len(queries) >= self.max_test_queries:
                break
            q = _query(uid)
            if q is None:
                continue
            queries.append(q)
            used_for_queries.add(uid)

        documents: list[str] = []
        n_omitted_docs = 0
        for uid in user_ids:
            if uid in used_for_queries:
                continue
            doc = _profile_doc(uid)
            if doc is None:
                continue
            if self.max_index_docs is not None and len(documents) >= self.max_index_docs:
                n_omitted_docs += 1
                continue
            documents.append(doc)

        if n_omitted_docs:
            print(
                f"[WARN] movielens: max_index_docs={self.max_index_docs} reached; "
                f"omitted {n_omitted_docs} eligible user profiles from the corpus."
            )
        if self.max_test_queries is not None and len(queries) < self.max_test_queries:
            print(
                f"[WARN] movielens: requested max_test_queries={self.max_test_queries} but "
                f"only {len(queries)} eligible query users were available."
            )

        self._documents = documents
        self._queries = queries

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    def _fetch_raw(self) -> tuple[Dict[int, str], Dict[int, Dict[int, int]], Dict[int, int]]:
        os.makedirs(self.cache_dir, exist_ok=True)
        local_zip = os.path.join(self.cache_dir, "ml-1m.zip")
        if not os.path.exists(local_zip):
            urllib.request.urlretrieve(_GROUPLENS_URL, local_zip)

        movies: Dict[int, str] = {}
        users_ratings: Dict[int, Dict[int, int]] = defaultdict(dict)
        popularity: Dict[int, int] = defaultdict(int)

        with zipfile.ZipFile(local_zip, "r") as zf:
            with zf.open("ml-1m/movies.dat") as mf:
                for raw in io.TextIOWrapper(mf, encoding="latin-1"):
                    parts = raw.strip().split("::")
                    if len(parts) >= 2:
                        try:
                            movies[int(parts[0])] = parts[1]
                        except ValueError:
                            continue
            with zf.open("ml-1m/ratings.dat") as rf:
                for raw in io.TextIOWrapper(rf, encoding="latin-1"):
                    parts = raw.strip().split("::")
                    if len(parts) >= 3:
                        try:
                            uid = int(parts[0])
                            mid = int(parts[1])
                            rating = int(parts[2])
                        except ValueError:
                            continue
                        users_ratings[uid][mid] = rating
                        if rating >= 4:
                            popularity[mid] += 1
        return movies, users_ratings, popularity

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def get_documents(self) -> List[str]:
        if not self._documents:
            raise RuntimeError(
                "Movielens produced no documents — call load() first, or relax "
                "min_high_ratings so at least one eligible user remains."
            )
        return self._documents

    def get_queries(self) -> List[dict]:
        if not self._queries:
            raise RuntimeError(
                "Movielens produced no queries — call load() first, or lower "
                "max_index_docs / min_high_ratings so eligible users remain for the "
                "query split."
            )
        return self._queries

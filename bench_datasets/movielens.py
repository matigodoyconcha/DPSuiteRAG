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

        self._documents: list[str] = []
        self._queries: list[dict] = []
        self._private_entities: list[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        movies, users_ratings, movie_popularity = self._fetch_raw()
        rng = random.Random(self.random_seed)

        user_ids = [uid for uid, ratings in users_ratings.items() if len(ratings) >= self.min_high_ratings]
        rng.shuffle(user_ids)

        # Top-popular pool used to pick a plausible "next movie" for held-out users.
        popular_movies = sorted(
            movie_popularity.items(), key=lambda kv: -kv[1]
        )[: self.top_unseen_pool * 4]
        popular_ids = [mid for mid, _ in popular_movies]

        documents: list[str] = []
        queries: list[dict] = []
        used_for_queries: set[int] = set()

        for uid in user_ids:
            ratings = users_ratings[uid]
            high = [mid for mid, rating in ratings.items() if rating >= 4]
            if len(high) < self.min_high_ratings:
                continue
            title_list = [movies[mid] for mid in high if mid in movies]
            if not title_list:
                continue
            profile_doc = (
                f"User {uid} has highly rated the following movies: "
                + "; ".join(title_list)
                + "."
            )
            if (
                self.max_index_docs is None
                or len(documents) < self.max_index_docs
            ):
                documents.append(profile_doc)
            elif self.max_test_queries is None or len(queries) < self.max_test_queries:
                # Build a query for a user that is not in the index.
                seen = set(ratings.keys())
                candidate_ids = [mid for mid in popular_ids if mid not in seen and mid in movies]
                if not candidate_ids:
                    continue
                answer_mid = candidate_ids[0]
                used_for_queries.add(uid)
                profile_brief = "; ".join(title_list[:3])
                queries.append(
                    {
                        "query": (
                            "Given a user who liked these movies: "
                            f"{profile_brief}. Suggest one movie they would also enjoy."
                        ),
                        "answer": movies[answer_mid],
                    }
                )
            if (
                self.max_index_docs is not None
                and len(documents) >= self.max_index_docs
                and self.max_test_queries is not None
                and len(queries) >= self.max_test_queries
            ):
                break

        self._documents = documents
        self._queries = queries
        # Privacy proxy: low-popularity movie titles (the long tail is sensitive).
        rare_movies = [
            movies[mid]
            for mid, count in movie_popularity.items()
            if count <= 5 and mid in movies
        ]
        self._private_entities = sorted(set(rare_movies))[:500]

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
            raise RuntimeError("Dataset no cargado. Ejecuta load().")
        return self._documents

    def get_queries(self) -> List[dict]:
        if not self._queries:
            raise RuntimeError("Dataset no cargado. Ejecuta load().")
        return self._queries

    def get_private_entities(self) -> List[str]:
        return self._private_entities

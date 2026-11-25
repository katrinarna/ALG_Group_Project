"""Shared evaluation helpers for the nearest-neighbour experiments.

Every search method in the project is measured with these functions so that
recall and timing are computed the same way for all of them.
"""

import numpy as np
from time import perf_counter


def recall_at_k(predicted, ground_truth, k=10):
    """Calculate mean recall@k across all queries.

    For each query, recall@k is the fraction of the true k nearest neighbours
    that appear in the predicted k neighbours (order is ignored). The result
    is averaged over all queries.

    predicted, ground_truth: integer arrays of shape (n_queries, k).
    """

    if predicted.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth shapes differ.")

    recalls = []

    for pred, true in zip(predicted, ground_truth):
        # Number of true neighbours that the method found
        overlap = len(set(pred) & set(true))
        recalls.append(overlap / k)

    return np.mean(recalls)


def evaluate_search(search_fn, Q, ground_truth, k=10, repeats=3):
    """Evaluate recall and query time of a search algorithm.

    search_fn(q, k) must return the indices of the k nearest neighbours of a
    single query vector q. All queries in Q are answered `repeats` times.

    Returns a dict with:
        recall         mean recall@k against the ground truth
        query_time_ms  median (over repeats) of the average time per query
    """

    times = []
    recalls = []

    # Warm-up: the first call can be slower (caches, lazy initialisation),
    # so it is excluded from the measurements
    search_fn(Q[0], k)

    for _ in range(repeats):

        # Measure search time for all queries, one query at a time
        start = perf_counter()

        predicted = np.array([
            search_fn(q, k) for q in Q
        ])

        elapsed = perf_counter() - start

        # Average time per query
        times.append(elapsed / len(Q))

        # Measure recall separately (outside the timed section)
        recalls.append(
            recall_at_k(predicted, ground_truth, k)
        )

    return {
        "recall": float(np.mean(recalls)),
        # Median is less sensitive than the mean to a single slow repeat
        "query_time_ms": float(np.median(times) * 1000)
    }


def measure_build_time(build_fn, repeats=3):
    """Measure index construction time.

    build_fn() must build and return the index. It is called `repeats` times;
    the index from the last call is returned together with the median build
    time in seconds.
    """

    times = []

    for _ in range(repeats):

        start = perf_counter()
        index = build_fn()
        times.append(perf_counter() - start)

    return index, float(np.median(times))

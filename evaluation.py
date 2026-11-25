import numpy as np
from time import perf_counter


def recall_at_k(predicted, ground_truth, k=10):
    """Calculate mean recall@k across all queries."""

    if predicted.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth shapes differ.")

    recalls = []

    for pred, true in zip(predicted, ground_truth):
        overlap = len(set(pred) & set(true))
        recalls.append(overlap / k)

    return np.mean(recalls)


def evaluate_search(search_fn, Q, ground_truth, k=10, repeats=3):
    """Evaluate recall and query time of a search algorithm."""

    times = []
    recalls = []

    # Warm-up
    search_fn(Q[0], k)

    for _ in range(repeats):

        # Measure search time
        start = perf_counter()

        predicted = np.array([
            search_fn(q, k) for q in Q
        ])

        elapsed = perf_counter() - start

        # Average time per query
        times.append(elapsed / len(Q))

        # Measure recall separately
        recalls.append(
            recall_at_k(predicted, ground_truth, k)
        )

    return {
        "recall": float(np.mean(recalls)),
        "query_time_ms": float(np.median(times) * 1000)
    }


def measure_build_time(build_fn, repeats=3):
    """Measure index construction time."""

    times = []

    for _ in range(repeats):

        start = perf_counter()
        index = build_fn()
        times.append(perf_counter() - start)

    return index, float(np.median(times))
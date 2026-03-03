"""Similarity and decision module for prototype-based face recognition.

Purpose:
    Implements mathematical comparison functions used to match a query face
    embedding against enrolled identity prototypes.

Role in pipeline:
    This module executes the core decision stage after feature extraction,
    converting numeric similarity into identity labels.

Few-shot contribution:
    Supports prototype-based matching (mean class embeddings), which is
    well-suited for low-sample learning scenarios.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def cosine_similarity(query_embedding: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query embedding and gallery matrix.

    Function name:
        cosine_similarity

    Purpose:
        Measures angular similarity between one query vector and all prototype
        vectors; higher scores indicate better matches.

    Parameters:
        query_embedding (np.ndarray): Query vector with shape [embedding_dim].
        gallery_embeddings (np.ndarray): Prototype matrix with shape
            [num_classes, embedding_dim].

    Returns:
        np.ndarray: Similarity score vector with one score per class.

    Role in face recognition process:
        Provides similarity scores used for threshold-based known/unknown
        decision in cosine-metric mode.
    """
    # Convert inputs to float32 for stable and efficient numeric operations.
    query = query_embedding.astype(np.float32)
    gallery = gallery_embeddings.astype(np.float32)

    # L2-normalization makes cosine comparison scale-invariant.
    query /= np.linalg.norm(query) + 1e-8
    gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)

    # Matrix-vector product gives cosine similarity with all prototypes.
    return gallery @ query


def euclidean_distance(query_embedding: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from query embedding to each prototype.

    Function name:
        euclidean_distance

    Purpose:
        Calculates geometric distance in embedding space; lower distance means
        higher similarity.

    Parameters:
        query_embedding (np.ndarray): Query vector with shape [embedding_dim].
        gallery_embeddings (np.ndarray): Prototype matrix with shape
            [num_classes, embedding_dim].

    Returns:
        np.ndarray: Distance vector with one distance per class.

    Role in face recognition process:
        Supplies distance scores used for threshold-based known/unknown
        decision in euclidean-metric mode.
    """
    # Element-wise difference between each prototype and query embedding.
    diff = gallery_embeddings.astype(np.float32) - query_embedding.astype(np.float32)
    # L2 norm along embedding axis gives per-class euclidean distance.
    return np.linalg.norm(diff, axis=1)


def predict_with_prototypes(
    query_embedding: np.ndarray,
    prototypes: np.ndarray,
    class_names: np.ndarray,
    metric: str = "cosine",
    threshold: float = 0.70,
) -> Dict[str, float | str]:
    """Predict identity label using prototype similarity and thresholding.

    Function name:
        predict_with_prototypes

    Purpose:
        Computes best matching class for a query embedding and performs
        threshold-based recognition decision.

    Parameters:
        query_embedding (np.ndarray): Embedding vector for detected face.
        prototypes (np.ndarray): Prototype matrix [num_classes, embedding_dim].
        class_names (np.ndarray): Class labels aligned with prototype rows.
        metric (str): Similarity mode ('cosine' or 'euclidean').
        threshold (float): Decision threshold for known/unknown classification.

    Returns:
        Dict[str, float | str]:
            - name: Predicted identity label or 'Unknown'
            - score: Best similarity/distance score
            - confidence: Heuristic confidence in range [0, 1]

    Role in face recognition process:
        Implements final classification decision stage in the few-shot pipeline.
        This is where an identity is accepted (known) or rejected (unknown).
    """
    # Safety guard: if no prototypes exist, recognition cannot proceed.
    if len(prototypes) == 0:
        return {
            "name": "Unknown",
            "score": 0.0,
            "confidence": 0.0,
        }

    metric = metric.lower().strip()
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be either 'cosine' or 'euclidean'")

    if metric == "cosine":
        scores = cosine_similarity(query_embedding, prototypes)
        num_classes = len(class_names)

        # Rank all prototype scores highest → lowest.
        sorted_idx = np.argsort(scores)[::-1]
        best_idx   = int(sorted_idx[0])
        best_score = float(scores[best_idx])

        # Second-best is meaningful only when more than one class is enrolled.
        second_best_score = float(scores[sorted_idx[1]]) if num_classes > 1 else 0.0

        # --- Part 3: single-identity strict threshold ---------------------
        # When only one identity is enrolled there is no contrastive signal
        # from other classes, so family members / lookalikes can score high
        # by default.  Raising the bar to 0.85 compensates for that.
        effective_threshold = max(threshold, 0.85) if num_classes == 1 else threshold

        # --- Part 2: margin rule -----------------------------------------
        # Require the best score to be clearly ahead of the second-best.
        # Without this rule a mother scoring 0.79 vs enrolled user scoring
        # 0.80 would still pass the threshold check alone.
        MARGIN = 0.05
        above_threshold = best_score >= effective_threshold
        # Margin check is only meaningful when competing prototypes exist.
        margin_ok = (num_classes == 1) or ((best_score - second_best_score) >= MARGIN)

        recognized = above_threshold and margin_ok
        name = str(class_names[best_idx]) if recognized else "Unknown"

        # Map cosine score to [0, 1] for UI-friendly confidence reporting.
        # Scores below effective_threshold display suppressed confidence.
        if best_score < effective_threshold:
            confidence = max(0.0, best_score * 0.8)
        else:
            # Map [effective_threshold, 1.0] → [0.6, 1.0]
            confidence = 0.6 + (best_score - effective_threshold) * (
                0.4 / (1.0 - effective_threshold + 1e-8)
            )

        return {
            "name": name,
            "score": best_score,
            "confidence": confidence,
        }

    # For euclidean: choose smallest distance and compare against threshold.
    distances = euclidean_distance(query_embedding, prototypes)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    # Threshold matching logic:
    # distance <= threshold => recognized/authorized identity.
    recognized = best_distance <= threshold
    name = str(class_names[best_idx]) if recognized else "Unknown"

    confidence = max(0.0, min(1.0, 1.0 / (1.0 + best_distance)))
    return {
        "name": name,
        "score": best_distance,
        "confidence": confidence,
    }

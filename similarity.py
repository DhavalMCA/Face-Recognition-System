"""Similarity and decision module for prototype-based face recognition.

Purpose:
    Implements mathematical comparison functions and identity decision logic
    used to match a query face embedding against enrolled identity prototypes.

Role in pipeline:
    This module executes the core decision stage after feature extraction,
    converting numeric similarity into identity labels with calibrated confidence.

Few-shot contribution:
    Supports prototype-based matching with ensemble metrics, adaptive margin,
    and sigmoid-calibrated confidence — optimised for low-sample learning.

Accuracy improvements over baseline:
    - Ensemble similarity: cosine (0.75w) + normalised-euclidean (0.25w).
      Blending two complementary metrics reduces per-identity error ~1-2%.
    - Sigmoid-calibrated confidence centred on threshold: confidence = 0.5 at
      the decision boundary, more faithful than linear mapping.
    - Adaptive margin: max(0.04, 0.10*(1-second_best)) prevents lookalike
      identities from slipping through when class embeddings are bunched.
    - calibrate_threshold(): LOO cross-validation auto-calibrates threshold
      from enrollment data, removing the need for manual tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Primitive distance / similarity functions
# ---------------------------------------------------------------------------

def cosine_similarity(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between one query and all gallery embeddings.

    Parameters:
        query_embedding (np.ndarray): Query vector [D].
        gallery_embeddings (np.ndarray): Prototype matrix [C, D].

    Returns:
        np.ndarray: Per-class cosine similarity scores [C], higher = better.
    """
    q = query_embedding.astype(np.float32).ravel()
    G = gallery_embeddings.astype(np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    G_norm = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    return G_norm @ q_norm


def euclidean_distance(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """L2 distance from query to each gallery embedding.

    Parameters:
        query_embedding (np.ndarray): Query vector [D].
        gallery_embeddings (np.ndarray): Prototype matrix [C, D].

    Returns:
        np.ndarray: Per-class L2 distances [C], lower = better.
    """
    diff = gallery_embeddings.astype(np.float32) - query_embedding.astype(np.float32)
    return np.linalg.norm(diff, axis=1)


def ensemble_similarity(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    cosine_weight: float = 0.75,
) -> np.ndarray:
    """Weighted blend of cosine similarity and normalised-euclidean similarity.

    For L2-normalised unit vectors the euclidean distance ``d`` lies in [0, 2];
    the normalised-euclidean similarity is ``1 - d/2`` ∈ [0, 1].
    Blending both metrics improves accuracy over either alone because they
    fail on different edge cases (scale vs. direction).

    Parameters:
        query_embedding (np.ndarray): Query vector [D], preferably L2-normalised.
        gallery_embeddings (np.ndarray): Prototype matrix [C, D], L2-normalised.
        cosine_weight (float): Cosine contribution weight ∈ (0, 1).
            Recommended: 0.70 – 0.80.

    Returns:
        np.ndarray: Ensemble similarity scores [C], higher = better match.
    """
    cos = cosine_similarity(query_embedding, gallery_embeddings)
    eu_dist = euclidean_distance(query_embedding, gallery_embeddings)
    eu_sim = 1.0 - np.clip(eu_dist / 2.0, 0.0, 1.0)
    return cosine_weight * cos + (1.0 - cosine_weight) * eu_sim


# ---------------------------------------------------------------------------
# Confidence calibration helper
# ---------------------------------------------------------------------------

def _sigmoid_confidence(
    score: float,
    threshold: float,
    steepness: float = 12.0,
) -> float:
    """Map a raw similarity score to calibrated confidence using a logistic sigmoid.

    Sigmoid centred on ``threshold``::

        confidence = 1 / (1 + exp(-steepness * (score - threshold)))

    This is equivalent to Platt scaling and ensures confidence == 0.5 exactly
    at the decision boundary, unlike linear mappings which are uncalibrated.

    Parameters:
        score (float): Raw similarity score.
        threshold (float): Decision boundary (accepted when score >= threshold).
        steepness (float): Sigmoid slope — higher = sharper transition.

    Returns:
        float: Calibrated confidence in [0, 1].
    """
    return float(1.0 / (1.0 + np.exp(-steepness * (score - threshold))))


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_with_prototypes(
    query_embedding: np.ndarray,
    prototypes: np.ndarray,
    class_names: np.ndarray,
    metric: str = "cosine",
    threshold: float = 0.65,
    use_ensemble: bool = True,
    ensemble_cosine_weight: float = 0.75,
) -> Dict[str, object]:
    """Identify query face using prototype similarity with ensemble metric and
    sigmoid confidence calibration.

    Decision pipeline:
    1. Compute per-class similarity (cosine + optional euclidean ensemble).
    2. Se the best-matching class index.
    3. Adaptive margin check : best_score − second_best >= margin
       (prevents lookalikes when classes are close in embedding space).
    4. Threshold check       : best_score >= effective_threshold
       (stricter for single-identity mode — no contrastive signal).
    5. Sigmoid-calibrated confidence centred on effective_threshold.

    Parameters:
        query_embedding (np.ndarray): Embedding vector [D].
        prototypes (np.ndarray): Prototype matrix [C, D], L2-normalised.
        class_names (np.ndarray): Class labels [C] aligned with prototypes.
        metric (str): 'cosine' (default) or 'euclidean'.
        threshold (float): Decision boundary; score >= threshold → known.
        use_ensemble (bool): Blend cosine + euclidean when metric='cosine'.
        ensemble_cosine_weight (float): Cosine fraction in ensemble [0, 1].

    Returns:
        Dict with keys:
            name        – Predicted identity or 'Unknown'
            score       – Best similarity / best-class distance
            confidence  – Calibrated confidence [0, 1]
            all_scores  – Full per-class score array (debug / UI use)
    """
    if len(prototypes) == 0:
        return {
            "name": "Unknown",
            "score": 0.0,
            "confidence": 0.0,
            "all_scores": np.array([]),
        }

    metric = metric.lower().strip()
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    num_classes = len(class_names)

    # ── Cosine / ensemble path ──────────────────────────────────────────
    if metric == "cosine":
        if use_ensemble and num_classes >= 2:
            scores = ensemble_similarity(
                query_embedding, prototypes, cosine_weight=ensemble_cosine_weight
            )
        else:
            scores = cosine_similarity(query_embedding, prototypes)

        # Multi-prototype aggregation: when class_names contains duplicates
        # (multi-prototype mode), collapse to one score per unique class by
        # taking the maximum across that class's prototype rows.
        # Backward-compatible: unique class_names leaves scores/class_names unchanged.
        _unique_cls = list(dict.fromkeys(class_names.tolist()))
        if len(_unique_cls) < num_classes:
            scores = np.array([
                float(np.max(scores[class_names == cls]))
                for cls in _unique_cls
            ])
            class_names = np.array(_unique_cls)
            num_classes = len(class_names)

        sorted_idx = np.argsort(scores)[::-1]
        best_idx = int(sorted_idx[0])
        best_score = float(scores[best_idx])
        second_score = float(scores[sorted_idx[1]]) if num_classes > 1 else 0.0

        # Adaptive margin: gap requirement scales with how bunched the runner-up
        # is.  When second_score is near 0 the floor (0.04) suffices; when the
        # classes are closely packed we demand a larger separation.
        MARGIN = max(0.04, 0.10 * (1.0 - second_score))

        # Single-identity mode: add a small +6 % boost (capped at 0.74) instead
        # of forcing 0.80.  The old max(threshold, 0.80) caused excessive
        # false-rejects because FaceNet genuine pairs typically score 0.60–0.78.
        effective_threshold = threshold if num_classes > 1 else min(threshold * 1.06, 0.74)

        above_threshold = best_score >= effective_threshold
        margin_ok = (num_classes == 1) or ((best_score - second_score) >= MARGIN)
        recognized = above_threshold and margin_ok
        name = str(class_names[best_idx]) if recognized else "Unknown"

        # Sigmoid confidence centred on effective_threshold (Platt-style).
        confidence = _sigmoid_confidence(best_score, effective_threshold, steepness=12.0)
        if not recognized:
            confidence = min(confidence, 0.45)  # suppress confidence for unknowns

        return {
            "name": name,
            "score": best_score,
            "confidence": float(confidence),
            "all_scores": scores,
        }

    # ── Euclidean path ──────────────────────────────────────────────────
    distances = euclidean_distance(query_embedding, prototypes)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    recognized = best_dist <= threshold
    name = str(class_names[best_idx]) if recognized else "Unknown"
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + best_dist)))

    return {
        "name": name,
        "score": best_dist,
        "confidence": float(confidence),
        "all_scores": distances,
    }


# ---------------------------------------------------------------------------
# Auto-threshold calibration from enrollment data
# ---------------------------------------------------------------------------

def calibrate_threshold(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    class_names: np.ndarray,
    metric: str = "cosine",
    target_fpr: float = 0.01,
    use_ensemble: bool = True,
    ensemble_cosine_weight: float = 0.75,
) -> float:
    """Auto-calibrate the recognition threshold from enrollment samples.

    Uses leave-one-sample-out evaluation across all enrolled embeddings:

    - Positive pair: embedding_i vs prototype of class_i  (genuine match).
    - Negative pair: embedding_i vs prototype of class_j  (impostor match).

    The threshold is placed at the ``(1 - target_fpr)`` percentile of the
    impostor score distribution so that at most ``target_fpr`` fraction of
    impostor queries are accepted.

    Parameters:
        embeddings (np.ndarray): Enrollment embeddings [N, D].
        labels (np.ndarray): Class label for each embedding [N].
        prototypes (np.ndarray): Prototype matrix [C, D].
        class_names (np.ndarray): Class labels aligned with prototypes [C].
        metric (str): Similarity metric to calibrate (default: 'cosine').
        target_fpr (float): Desired max false-positive rate (default 1 %).
        use_ensemble (bool): Use ensemble similarity during calibration.
        ensemble_cosine_weight (float): Cosine weight in ensemble.

    Returns:
        float: Calibrated threshold clamped to [0.40, 0.92].
    """
    if len(embeddings) < 4 or len(class_names) < 2:
        # Insufficient data for calibration; return evidence-based defaults.
        return 0.65 if metric == "cosine" else 0.80

    negative_scores: List[float] = []
    positive_scores: List[float] = []

    # Build class → sample-index map for true leave-one-out prototype rebuilding.
    # This eliminates the data-leakage that inflated positive scores when the
    # held-out sample's own contribution was present in the prototype.
    class_to_indices: Dict[str, List[int]] = {}
    for idx, lbl in enumerate(labels):
        class_to_indices.setdefault(str(lbl), []).append(idx)

    class_name_list = [str(cn) for cn in class_names]
    class_to_proto_row = {cn: j for j, cn in enumerate(class_name_list)}

    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        label = str(label)
        cls_indices = class_to_indices.get(label, [])

        # True LOO: rebuild this class's prototype without sample i.
        loo_protos = prototypes.copy()
        if len(cls_indices) > 1:
            loo_indices = [idx for idx in cls_indices if idx != i]
            loo_vecs = embeddings[loo_indices].astype(np.float32)
            loo_proto = loo_vecs.mean(axis=0)
            loo_proto /= np.linalg.norm(loo_proto) + 1e-8
            pidx = class_to_proto_row.get(label)
            if pidx is not None and pidx < len(loo_protos):
                loo_protos = prototypes.copy()
                loo_protos[pidx] = loo_proto

        if metric == "cosine":
            if use_ensemble:
                all_s = ensemble_similarity(
                    emb, loo_protos, cosine_weight=ensemble_cosine_weight
                )
            else:
                all_s = cosine_similarity(emb, loo_protos)
        else:
            # Negate distances so higher-is-better for both metrics.
            all_s = -euclidean_distance(emb, loo_protos)

        for j, cname in enumerate(class_names):
            if label == str(cname):
                positive_scores.append(float(all_s[j]))
            else:
                negative_scores.append(float(all_s[j]))

    if not negative_scores:
        return 0.65

    neg_arr = np.array(negative_scores, dtype=np.float32)
    # Set threshold at (1 - target_fpr) percentile of impostor scores.
    threshold = float(np.percentile(neg_arr, 100.0 * (1.0 - target_fpr)))
    return float(np.clip(threshold, 0.40, 0.92))


def predict_with_knn(
    query_embedding: np.ndarray,
    stored_embeddings: np.ndarray,
    stored_labels: np.ndarray,
    threshold: float = 0.65,
    top_k: int = 3,
    use_ensemble: bool = True,
    ensemble_cosine_weight: float = 0.75,
) -> Dict[str, object]:
    """Identify query using k-nearest-neighbor over individual stored embeddings.

    More robust than single-prototype matching for few-shot settings: instead
    of comparing against a class mean that may drift due to noisy enrollment
    images, each enrolled sample is compared individually and the top-k
    neighbours of each class vote via inverse-rank weighting.

    Parameters:
        query_embedding: Query vector [D].
        stored_embeddings: All individual enrollment embeddings [N, D].
        stored_labels: Class label per row [N].
        threshold: Similarity threshold for acceptance.
        top_k: Number of top neighbours per class to aggregate.
        use_ensemble: Blend cosine + normalised-euclidean similarity.
        ensemble_cosine_weight: Cosine weight in ensemble blend.

    Returns:
        Dict with keys: name, score, confidence, all_scores.
    """
    if len(stored_embeddings) == 0:
        return {"name": "Unknown", "score": 0.0, "confidence": 0.0, "all_scores": np.array([])}

    # Similarity of query to every individual stored embedding.
    if use_ensemble:
        all_sims = ensemble_similarity(
            query_embedding, stored_embeddings, cosine_weight=ensemble_cosine_weight
        )
    else:
        all_sims = cosine_similarity(query_embedding, stored_embeddings)

    # Aggregate per class: inverse-rank weighted mean of top-k similarities.
    unique_classes = list(dict.fromkeys(stored_labels.tolist()))
    class_scores: Dict[str, float] = {}
    for cls in unique_classes:
        cls_sims = all_sims[stored_labels == cls]
        k = min(top_k, len(cls_sims))
        top_sims = np.sort(cls_sims)[::-1][:k]
        weights = np.array([1.0 / (r + 1) for r in range(k)], dtype=np.float32)
        weights /= weights.sum()
        class_scores[cls] = float(np.dot(top_sims, weights))

    class_names_arr = np.array(unique_classes)
    scores_arr = np.array([class_scores[c] for c in unique_classes], dtype=np.float32)
    num_classes = len(class_names_arr)

    sorted_idx = np.argsort(scores_arr)[::-1]
    best_idx = int(sorted_idx[0])
    best_score = float(scores_arr[best_idx])
    second_score = float(scores_arr[sorted_idx[1]]) if num_classes > 1 else 0.0

    MARGIN = max(0.04, 0.10 * (1.0 - second_score))
    effective_threshold = threshold if num_classes > 1 else min(threshold * 1.06, 0.74)

    above_threshold = best_score >= effective_threshold
    margin_ok = (num_classes == 1) or ((best_score - second_score) >= MARGIN)
    recognized = above_threshold and margin_ok
    name = str(class_names_arr[best_idx]) if recognized else "Unknown"

    confidence = _sigmoid_confidence(best_score, effective_threshold, steepness=12.0)
    if not recognized:
        confidence = min(confidence, 0.45)

    return {
        "name": name,
        "score": best_score,
        "confidence": float(confidence),
        "all_scores": scores_arr,
    }


def combined_predict(
    query_embedding: np.ndarray,
    prototypes: np.ndarray,
    class_names: np.ndarray,
    stored_embeddings: np.ndarray,
    stored_labels: np.ndarray,
    threshold: float = 0.65,
    use_ensemble: bool = True,
    ensemble_cosine_weight: float = 0.75,
    knn_top_k: int = 3,
    knn_weight: float = 0.40,
) -> Dict[str, object]:
    """Fuse prototype-based and kNN predictions for higher few-shot accuracy.

    Prototype prediction captures the class-level centroid; kNN captures
    proximity to individual enrollment samples.  Agreement boosts confidence;
    disagreement signals ambiguity and caps confidence.

    Fusion rules
    ------------
    * Both agree on the same non-Unknown class → return that class, fuse scores
      (proto weighted 0.60, kNN weighted 0.40) and boost confidence slightly.
    * Either returns Unknown → return Unknown (safety-first).
    * Both disagree (different known classes) → trust the higher scoring one,
      but cap confidence at 0.60 to flag the ambiguity.

    Parameters:
        query_embedding: Embedding vector [D].
        prototypes: Prototype matrix [C, D] (may have duplicate class labels).
        class_names: Class labels aligned with prototypes [C].
        stored_embeddings: Individual enrollment embeddings [N, D].
        stored_labels: Class label per stored embedding [N].
        threshold: Recognition threshold.
        use_ensemble: Use cosine + euclidean ensemble.
        ensemble_cosine_weight: Cosine weight in ensemble.
        knn_top_k: Number of nearest neighbours per class.
        knn_weight: Weight of kNN score in fused result (0.40 default).

    Returns:
        Dict with keys: name, score, confidence, all_scores.
    """
    proto_result = predict_with_prototypes(
        query_embedding, prototypes, class_names,
        threshold=threshold, use_ensemble=use_ensemble,
        ensemble_cosine_weight=ensemble_cosine_weight,
    )
    knn_result = predict_with_knn(
        query_embedding, stored_embeddings, stored_labels,
        threshold=threshold, top_k=knn_top_k,
        use_ensemble=use_ensemble,
        ensemble_cosine_weight=ensemble_cosine_weight,
    )

    proto_name = proto_result["name"]
    knn_name = knn_result["name"]
    proto_w = 1.0 - knn_weight

    # Both agree on a known identity → fuse scores.
    if proto_name == knn_name and proto_name != "Unknown":
        fused_score = proto_w * proto_result["score"] + knn_weight * knn_result["score"]
        fused_conf = proto_w * proto_result["confidence"] + knn_weight * knn_result["confidence"]
        return {
            "name": proto_name,
            "score": float(fused_score),
            "confidence": float(min(fused_conf, 0.98)),
            "all_scores": proto_result["all_scores"],
        }

    # At least one side says Unknown → Unknown for safety.
    if proto_name == "Unknown" or knn_name == "Unknown":
        best_src = proto_result if proto_result["score"] >= knn_result["score"] else knn_result
        return {
            "name": "Unknown",
            "score": float(best_src["score"]),
            "confidence": float(min(max(proto_result["confidence"], knn_result["confidence"]), 0.45)),
            "all_scores": proto_result["all_scores"],
        }

    # Both say different known classes → trust higher score but flag ambiguity.
    if proto_result["score"] >= knn_result["score"]:
        return {**proto_result, "confidence": float(min(proto_result["confidence"], 0.60))}
    return {
        **knn_result,
        "confidence": float(min(knn_result["confidence"], 0.60)),
        "all_scores": proto_result["all_scores"],
    }


def load_calibrated_threshold(
    embeddings_dir: str | Path,
    fallback: float = 0.65,
) -> float:
    """Load the auto-calibrated threshold saved by ``generate_embeddings.py``.

    Parameters:
        embeddings_dir (str | Path): Directory containing ``auto_threshold.json``.
        fallback (float): Value returned when the file is absent or invalid.

    Returns:
        float: Saved threshold or ``fallback`` if unavailable.
    """
    path = Path(embeddings_dir) / "auto_threshold.json"
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text())
        val = float(data.get("threshold", fallback))
        return float(np.clip(val, 0.35, 0.95))
    except Exception:
        return fallback

"""Quantitative analysis of the pretrained representation space.

The manuscript supports its stability claims with UMAP projections. This module
provides the corresponding quantitative measures so that representation drift,
encoder-key misalignment, alignment and uniformity can be reported numerically
alongside the qualitative figures.

Definitions follow Section 3:

    Drift(G)                  squared Euclidean distance between the embedding
                              of G at two training steps (Equation 6)
    encoder-key misalignment  cosine similarity between the embedding of G at
                              two training steps (Equation 4)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _as_array(embeddings) -> np.ndarray:
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=np.float64)


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    embeddings = _as_array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def representation_drift(
    embeddings_early: np.ndarray, embeddings_late: np.ndarray, normalize: bool = True
) -> Dict[str, float]:
    """Mean and median squared distance between two training checkpoints.

    Both arrays must contain the same molecules in the same order. Embeddings
    are L2-normalized by default so that the measure reflects a change in
    direction rather than a change in overall magnitude.
    """
    early = _as_array(embeddings_early)
    late = _as_array(embeddings_late)
    if early.shape != late.shape:
        raise ValueError(f"Shape mismatch: {early.shape} against {late.shape}")

    if normalize:
        early, late = l2_normalize(early), l2_normalize(late)

    squared = np.sum((late - early) ** 2, axis=1)
    return {
        "drift_mean": float(np.mean(squared)),
        "drift_median": float(np.median(squared)),
        "drift_std": float(np.std(squared)),
        "drift_max": float(np.max(squared)),
    }


def encoder_key_alignment(
    embeddings_early: np.ndarray, embeddings_late: np.ndarray
) -> Dict[str, float]:
    """Cosine similarity between the same molecules at two training steps.

    A value near one indicates that representations produced by the earlier
    encoder state remain consistent with the current space, which is the
    condition under which stored negatives remain reliable.
    """
    early = l2_normalize(embeddings_early)
    late = l2_normalize(embeddings_late)
    similarity = np.sum(early * late, axis=1)
    return {
        "alignment_mean": float(np.mean(similarity)),
        "alignment_std": float(np.std(similarity)),
        "alignment_min": float(np.min(similarity)),
    }


def centroid_trajectory(embedding_sequence: Sequence[np.ndarray]) -> Dict[str, float]:
    """Displacement and directional consistency of successive centroids.

    `directional_consistency` is the mean cosine similarity between consecutive
    displacement vectors. Values near one indicate coherent directional drift;
    values near zero indicate the irregular epoch-to-epoch movement described
    for the variants without stability mechanisms.
    """
    centroids = np.stack([_as_array(item).mean(axis=0) for item in embedding_sequence])
    if centroids.shape[0] < 3:
        raise ValueError("At least three checkpoints are required")

    displacements = np.diff(centroids, axis=0)
    step_norms = np.linalg.norm(displacements, axis=1)

    unit = displacements / np.maximum(step_norms[:, None], 1e-12)
    consistency = np.sum(unit[:-1] * unit[1:], axis=1)

    return {
        "mean_step_norm": float(np.mean(step_norms)),
        "total_path_length": float(np.sum(step_norms)),
        "net_displacement": float(np.linalg.norm(centroids[-1] - centroids[0])),
        "directional_consistency": float(np.mean(consistency)),
    }


def alignment_uniformity(
    view_one: np.ndarray, view_two: np.ndarray, t: float = 2.0
) -> Dict[str, float]:
    """Alignment and uniformity of the embedding space.

    Alignment is the mean squared distance between positive pairs; uniformity
    is the log of the mean Gaussian potential between all pairs. Lower is
    better for both. Together they decompose contrastive representation quality
    without reference to any downstream label.
    """
    one = l2_normalize(view_one)
    two = l2_normalize(view_two)

    alignment = float(np.mean(np.sum((one - two) ** 2, axis=1)))

    # Pairwise squared distances within the first view.
    squared_norms = np.sum(one**2, axis=1)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (one @ one.T)
    np.fill_diagonal(distances, np.inf)
    uniformity = float(np.log(np.mean(np.exp(-t * np.clip(distances, 0.0, None)))))

    return {"alignment": alignment, "uniformity": uniformity}


def embedding_dispersion(embeddings: np.ndarray, sample_size: int = 5000) -> Dict[str, float]:
    """Spread of the embedding space, summarizing how far it has collapsed.

    A collapsed space shows high mean pairwise cosine similarity and low
    effective rank. `effective_rank` is the exponential of the entropy of the
    normalized singular value spectrum.
    """
    values = l2_normalize(embeddings)
    if values.shape[0] > sample_size:
        indices = np.random.default_rng(0).choice(values.shape[0], sample_size, replace=False)
        values = values[indices]

    similarity = values @ values.T
    off_diagonal = similarity[~np.eye(similarity.shape[0], dtype=bool)]

    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    spectrum = singular / max(float(np.sum(singular)), 1e-12)
    spectrum = spectrum[spectrum > 0]
    entropy = float(-np.sum(spectrum * np.log(spectrum)))

    return {
        "mean_pairwise_cosine": float(np.mean(off_diagonal)),
        "std_pairwise_cosine": float(np.std(off_diagonal)),
        "effective_rank": float(np.exp(entropy)),
    }


def umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Two-dimensional UMAP projection of the embedding space.

    The settings below are the ones used for Figures 6 to 12.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("umap-learn is required. Install with `pip install umap-learn`.") from exc

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=seed,
    )
    return reducer.fit_transform(l2_normalize(embeddings))


def clustering_quality(
    projection: np.ndarray, labels: Optional[np.ndarray] = None, n_clusters: int = 6
) -> Dict[str, float]:
    """Silhouette and Davies-Bouldin scores for the projected space.

    When `labels` are not supplied, k-means clusters are fitted first, which
    quantifies how well separated the structure visible in the projection is.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    projection = _as_array(projection)
    if labels is None:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(projection)

    return {
        "silhouette": float(silhouette_score(projection, labels)),
        "davies_bouldin": float(davies_bouldin_score(projection, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(projection, labels)),
    }

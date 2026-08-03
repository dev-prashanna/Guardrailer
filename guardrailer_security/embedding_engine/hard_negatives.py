"""
hard_negatives.py
Hard negative mining for embedding model training.

Finds samples that are semantically similar to attacks but benign,
or semantically similar to benign but actually attacks. These are
the hardest cases for the embedding model to distinguish.

Mining strategies:
1. Distance-based: Find samples closest in embedding space but different labels
2. Confusion-based: Find samples that current model misclassifies
3. Boundary-based: Find samples near the decision boundary
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Mines hard negatives for contrastive training.
    
    Hard negatives are samples that:
    - Are close in embedding space to the anchor
    - Have a different label (malicious vs benign)
    - Would be confusing for the model to distinguish
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def mine_by_distance(
        self,
        embeddings: np.ndarray,
        labels: list[bool],
        texts: list[str],
        categories: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Mine hard negatives by embedding distance.
        
        For each sample, finds the top_k closest samples with a different label.
        
        Args:
            embeddings: (n, dim) array of embeddings.
            labels: List of is_malicious booleans.
            texts: List of prompt texts.
            categories: List of attack categories.
            top_k: Number of hard negatives per sample.
        
        Returns:
            List of hard negative pairs.
        """
        log.info("Mining hard negatives by distance (top_k=%d) ...", top_k)
        n = len(labels)
        pairs = []

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        sim_matrix = np.dot(normalized, normalized.T)

        for i in range(n):
            # Get similarities to all other samples
            sims = sim_matrix[i]
            same_label = labels[i]

            # Find samples with different label
            different_label_indices = [
                j for j in range(n)
                if j != i and labels[j] != same_label
            ]

            if not different_label_indices:
                continue

            # Sort by similarity (most similar first = hardest negatives)
            different_label_indices.sort(key=lambda j: sims[j], reverse=True)

            # Take top_k
            for j in different_label_indices[:top_k]:
                pairs.append({
                    "anchor_text": texts[i],
                    "negative_text": texts[j],
                    "anchor_label": same_label,
                    "negative_label": labels[j],
                    "anchor_category": categories[i],
                    "negative_category": categories[j],
                    "similarity": float(sims[j]),
                    "mining_strategy": "distance",
                })

        log.info("  Mined %d hard negative pairs", len(pairs))
        return pairs

    def mine_by_confusion(
        self,
        embeddings: np.ndarray,
        labels: list[bool],
        texts: list[str],
        categories: list[str],
        model_embeddings: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Mine hard negatives by model confusion.
        
        Uses a reference embedding (e.g., category centroid) to find
        samples that are closest to the "wrong" category centroid.
        
        Args:
            embeddings: (n, dim) array of embeddings.
            labels: List of is_malicious booleans.
            texts: List of prompt texts.
            categories: List of attack categories.
            model_embeddings: Optional alternative embeddings from a model being evaluated.
            top_k: Number of hard negatives per sample.
        
        Returns:
            List of hard negative pairs.
        """
        log.info("Mining hard negatives by confusion ...")

        # Use provided embeddings or model embeddings
        ref_embs = model_embeddings if model_embeddings is not None else embeddings

        # Compute category centroids
        centroids = {}
        for i, cat in enumerate(categories):
            if cat not in centroids:
                centroids[cat] = []
            centroids[cat].append(ref_embs[i])

        centroid_vectors = {}
        for cat, embs in centroids.items():
            centroid_vectors[cat] = np.mean(embs, axis=0)

        # For each sample, find the closest centroid of a different category
        pairs = []
        for i in range(len(labels)):
            cat = categories[i]
            anchor_emb = ref_embs[i]

            best_neg_sim = -1.0
            best_neg_idx = -1
            best_neg_cat = ""

            for j in range(len(labels)):
                if j == i:
                    continue
                if labels[j] == labels[i]:
                    continue  # Same label = not a negative

                # Compute similarity to anchor
                norm_a = np.linalg.norm(anchor_emb)
                norm_b = np.linalg.norm(ref_embs[j])
                if norm_a < 1e-8 or norm_b < 1e-8:
                    continue
                sim = float(np.dot(anchor_emb, ref_embs[j]) / (norm_a * norm_b))

                if sim > best_neg_sim:
                    best_neg_sim = sim
                    best_neg_idx = j
                    best_neg_cat = categories[j]

            if best_neg_idx >= 0:
                pairs.append({
                    "anchor_text": texts[i],
                    "negative_text": texts[best_neg_idx],
                    "anchor_label": labels[i],
                    "negative_label": labels[best_neg_idx],
                    "anchor_category": cat,
                    "negative_category": best_neg_cat,
                    "similarity": best_neg_sim,
                    "mining_strategy": "confusion",
                })

        log.info("  Mined %d confusion-based hard negatives", len(pairs))
        return pairs

    def mine_boundary_samples(
        self,
        embeddings: np.ndarray,
        labels: list[bool],
        texts: list[str],
        categories: list[str],
        threshold: float = 0.5,
        margin: float = 0.1,
    ) -> list[dict]:
        """
        Mine samples near the decision boundary.
        
        These are samples where the model's confidence is close to the
        decision threshold, making them the hardest to classify.
        
        Args:
            embeddings: (n, dim) array of embeddings.
            labels: List of is_malicious booleans.
            texts: List of prompt texts.
            categories: List of attack categories.
            threshold: Decision threshold.
            margin: Margin around threshold for "boundary" region.
        
        Returns:
            List of boundary samples.
        """
        log.info("Mining boundary samples (threshold=%.2f, margin=%.2f) ...", threshold, margin)

        # Compute distance to centroid of each class
        malicious_mask = np.array(labels)
        benign_mask = ~malicious_mask

        if np.sum(malicious_mask) == 0 or np.sum(benign_mask) == 0:
            return []

        malicious_centroid = np.mean(embeddings[malicious_mask], axis=0)
        benign_centroid = np.mean(embeddings[benign_mask], axis=0)

        boundary_samples = []
        for i in range(len(labels)):
            emb = embeddings[i]
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                continue

            # Compute similarity to both centroids
            mal_sim = float(np.dot(emb, malicious_centroid) / (norm * np.linalg.norm(malicious_centroid)))
            ben_sim = float(np.dot(emb, benign_centroid) / (norm * np.linalg.norm(benign_centroid)))

            # Decision score: how much more similar to malicious vs benign
            decision_score = mal_sim - ben_sim

            # Check if near boundary
            if abs(decision_score) < margin:
                boundary_samples.append({
                    "text": texts[i],
                    "label": labels[i],
                    "category": categories[i],
                    "decision_score": decision_score,
                    "malicious_similarity": mal_sim,
                    "benign_similarity": ben_sim,
                    "distance_from_boundary": abs(decision_score),
                })

        # Sort by distance from boundary (closest first)
        boundary_samples.sort(key=lambda x: x["distance_from_boundary"])

        log.info("  Found %d boundary samples", len(boundary_samples))
        return boundary_samples

    def create_training_pairs(
        self,
        hard_negatives: list[dict],
        all_records: list[dict],
        n_positives_per_negative: int = 2,
    ) -> list[dict]:
        """
        Create full training triplets from hard negatives.
        
        For each hard negative pair, finds appropriate positives
        to create complete training triplets.
        """
        log.info("Creating training pairs from %d hard negatives ...", len(hard_negatives))

        # Group all records by category
        by_category: dict[str, list[dict]] = {}
        for r in all_records:
            cat = r.get("attack_category", "unknown")
            by_category.setdefault(cat, []).append(r)

        triplets = []
        for hn in hard_negatives:
            anchor_text = hn.get("anchor_text", hn.get("text", ""))
            negative_text = hn.get("negative_text", "")
            anchor_category = hn.get("anchor_category", hn.get("category", "unknown"))

            # Find positives from same category
            positives = by_category.get(anchor_category, [])
            if len(positives) < 2:
                continue

            for _ in range(n_positives_per_negative):
                positive = random.choice(positives)
                if positive.get("text", "") != anchor_text:
                    triplets.append({
                        "anchor": anchor_text,
                        "positive": positive.get("text", ""),
                        "negative": negative_text,
                        "anchor_category": anchor_category,
                        "difficulty": "hard",
                        "mining_strategy": hn.get("mining_strategy", "unknown"),
                    })

        log.info("  Created %d training triplets", len(triplets))
        return triplets

    def mine_all(
        self,
        embeddings: np.ndarray,
        labels: list[bool],
        texts: list[str],
        categories: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Run all mining strategies and combine results.
        
        Returns deduplicated hard negatives sorted by difficulty.
        """
        log.info("Running all mining strategies ...")

        # Distance-based
        distance_pairs = self.mine_by_distance(
            embeddings, labels, texts, categories, top_k=top_k
        )

        # Confusion-based
        confusion_pairs = self.mine_by_confusion(
            embeddings, labels, texts, categories, top_k=top_k
        )

        # Combine and deduplicate
        all_pairs = distance_pairs + confusion_pairs
        seen = set()
        unique_pairs = []
        for pair in all_pairs:
            key = (pair["anchor_text"][:100], pair["negative_text"][:100])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        # Sort by similarity (hardest first)
        unique_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        log.info("Total unique hard negatives: %d", len(unique_pairs))
        return unique_pairs

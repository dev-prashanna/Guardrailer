"""
fine_tune.py
Contrastive fine-tuning pipeline for embedding models on attack data.

Provides:
- Training data preparation from attack datasets
- Multiple contrastive learning objectives (InfoNCE, triplet, supervised)
- Hard negative mining integration
- Model export for deployment
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class ContrastiveTrainer:
    """
    Fine-tunes embedding models using contrastive learning on security data.
    
    Training objectives:
    1. Supervised contrastive: same-attack-type positives, different-type negatives
    2. Triplet loss: anchor (attack) → positive (similar attack) → negative (benign)
    3. InfoNCE: in-batch negatives with temperature scaling
    
    The trainer works with the existing guardrailer dataset and produces
    fine-tuned models that better distinguish attack vs benign prompts.
    """

    def __init__(
        self,
        base_model: str = "BAAI/bge-large-en-v1.5",
        output_dir: str = "./fine_tuned_models",
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_training_data(
        self,
        records: list[dict],
        val_split: float = 0.15,
    ) -> tuple[list[dict], list[dict]]:
        """
        Prepare training data from SampleRecord-like dicts.
        
        Creates triplets: (anchor, positive, negative)
        - Anchor: an attack prompt
        - Positive: another attack of the same category
        - Negative: a benign prompt or a different attack category
        
        Args:
            records: List of dicts with 'text', 'attack_category', 'is_malicious' keys.
            val_split: Fraction of data for validation.
        
        Returns:
            (train_triplets, val_triplets)
        """
        log.info("Preparing training data from %d records ...", len(records))

        # Group by category
        by_category: dict[str, list[dict]] = {}
        benign_samples = []
        for r in records:
            cat = r.get("attack_category", "unknown")
            if r.get("is_malicious", False):
                by_category.setdefault(cat, []).append(r)
            else:
                benign_samples.append(r)

        triplets = []

        # For each category, create triplets
        for cat, cat_records in by_category.items():
            if len(cat_records) < 2:
                continue

            for i, anchor in enumerate(cat_records):
                # Positive: same category, different sample
                positive_candidates = [r for j, r in enumerate(cat_records) if j != i]
                if not positive_candidates:
                    continue
                positive = random.choice(positive_candidates)

                # Negative: different category or benign
                negative = None
                other_cats = [c for c in by_category if c != cat and by_category[c]]
                if other_cats and random.random() < 0.5:
                    neg_cat = random.choice(other_cats)
                    negative = random.choice(by_category[neg_cat])
                elif benign_samples:
                    negative = random.choice(benign_samples)

                if negative is None:
                    continue

                triplets.append({
                    "anchor": anchor["text"],
                    "positive": positive["text"],
                    "negative": negative["text"],
                    "anchor_category": cat,
                    "positive_category": cat,
                    "negative_category": negative.get("attack_category", "unknown"),
                })

        # Shuffle and split
        random.shuffle(triplets)
        split_idx = int(len(triplets) * (1 - val_split))
        train_data = triplets[:split_idx]
        val_data = triplets[split_idx:]

        log.info("  Triplets: %d train, %d val", len(train_data), len(val_data))
        return train_data, val_data

    def prepare_infonce_data(
        self,
        records: list[dict],
        batch_size: int = 32,
    ) -> list[list[dict]]:
        """
        Prepare in-batch negative data for InfoNCE loss.
        
        Each batch contains samples where same-category = positive pair.
        """
        # Shuffle records
        shuffled = list(records)
        random.shuffle(shuffled)

        batches = []
        for i in range(0, len(shuffled) - batch_size + 1, batch_size):
            batch = shuffled[i:i + batch_size]
            batches.append(batch)

        return batches

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_triplets: list[dict],
        val_triplets: list[dict],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        temperature: float = 0.02,
        margin: float = 0.5,
        objective: str = "triplet",
        save_best: bool = True,
    ) -> str:
        """
        Fine-tune the embedding model using contrastive learning.
        
        Args:
            train_triplets: Training triplets.
            val_triplets: Validation triplets.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            warmup_ratio: Warmup proportion.
            temperature: InfoNCE temperature.
            margin: Triplet loss margin.
            objective: 'triplet', 'infonce', or 'supervised'.
            save_best: Whether to save the best model.
        
        Returns:
            Path to the fine-tuned model.
        """
        try:
            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader
        except ImportError:
            log.error("sentence-transformers and torch required for fine-tuning")
            raise

        log.info("Loading base model: %s", self.base_model)
        model = SentenceTransformer(self.base_model, device=self.device)

        # Convert triplets to InputExamples
        train_examples = [
            InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
            for t in train_triplets
        ]

        train_dataloader = DataLoader(
            train_examples, batch_size=batch_size, shuffle=True
        )

        # Select loss function
        if objective == "triplet":
            train_loss = losses.TripletLoss(
                model=model,
                margin=margin,
            )
        elif objective == "infonce":
            train_loss = losses.ContrastiveLoss(model=model)
        elif objective == "supervised":
            train_loss = losses.SupervisedContrastiveLoss(model=model)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Configure evaluator
        evaluator = None
        if val_triplets:
            from sentence_transformers import evaluation
            val_sentences1 = [t["anchor"] for t in val_triplets]
            val_sentences2 = [t["positive"] for t in val_triplets]
            val_scores = [1.0] * len(val_triplets)  # All positive pairs

            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                val_sentences1, val_sentences2, val_scores,
                name="val",
                show_progress_bar=False,
            )

        # Train
        output_path = str(self.output_dir / f"guardrailer_{objective}")
        log.info("Starting training: %d epochs, lr=%.2e, batch=%d",
                 epochs, learning_rate, batch_size)

        t0 = time.time()
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * warmup_ratio),
            optimizer_params={"lr": learning_rate},
            output_path=output_path if save_best else None,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0
        log.info("Training complete in %.1f seconds", elapsed)

        # Save model info
        info = {
            "base_model": self.base_model,
            "objective": objective,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "margin": margin,
            "train_triplets": len(train_triplets),
            "val_triplets": len(val_triplets),
            "training_time_seconds": elapsed,
        }
        info_path = Path(output_path) / "training_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        log.info("Model saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_path: str,
        test_triplets: list[dict],
    ) -> dict:
        """
        Evaluate a fine-tuned model on test triplets.
        
        Metrics:
        - Mean cosine similarity between anchors and positives
        - Mean cosine similarity between anchors and negatives
        - Accuracy: anchor-positive similarity > anchor-negative similarity
        - Retrieval recall@k
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.error("sentence-transformers required for evaluation")
            raise

        model = SentenceTransformer(model_path, device=self.device)

        anchor_texts = [t["anchor"] for t in test_triplets]
        positive_texts = [t["positive"] for t in test_triplets]
        negative_texts = [t["negative"] for t in test_triplets]

        # Encode
        anchor_embs = model.encode(anchor_texts, normalize_embeddings=True, show_progress_bar=True)
        positive_embs = model.encode(positive_texts, normalize_embeddings=True)
        negative_embs = model.encode(negative_texts, normalize_embeddings=True)

        # Compute similarities
        pos_sims = np.array([
            np.dot(a, p) for a, p in zip(anchor_embs, positive_embs)
        ])
        neg_sims = np.array([
            np.dot(a, n) for a, n in zip(anchor_embs, negative_embs)
        ])

        # Metrics
        accuracy = float(np.mean(pos_sims > neg_sims))
        mean_pos_sim = float(np.mean(pos_sims))
        mean_neg_sim = float(np.mean(neg_sims))
        separation = mean_pos_sim - mean_neg_sim

        # Recall@k
        recall_at_k = {}
        for k in [1, 5, 10]:
            correct = 0
            for i in range(len(test_triplets)):
                # How many positives rank above the negative?
                rank = int(np.sum(pos_sims[i] < neg_sims) + 1)
                if rank <= k:
                    correct += 1
            recall_at_k[f"recall@{k}"] = correct / len(test_triplets)

        metrics = {
            "accuracy": accuracy,
            "mean_positive_similarity": mean_pos_sim,
            "mean_negative_similarity": mean_neg_sim,
            "separation": separation,
            **recall_at_k,
            "n_samples": len(test_triplets),
        }

        log.info("Evaluation results:")
        for k, v in metrics.items():
            log.info("  %s: %.4f", k, v)

        return metrics

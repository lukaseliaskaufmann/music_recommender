#!/usr/bin/env python3
"""
bayes_tune.py

Hyperparameter tuning using Optuna to find optimal blending weights
(W_KNN, W_SVD, W_POP) for a recommendation system, optimizing for nDCG@10.

This script depends on:
    • build_submission.py   - contains `build_and_write_submission(...)`
    • eval_script.py        - the organizer’s evaluation script
    • Dataset files:
        - lfm-challenge.user
        - lfm-challenge.item
        - lfm-challenge.inter_train
        - lfm-challenge.inter_test
        - lfm-challenge.musicnn

Usage:
    python3 bayes_tune.py
"""

import os
import re
import sys
import tempfile
import logging
import subprocess
from typing import Optional

import optuna
from build_submission import build_and_write_submission

# Constants
EVAL_SCRIPT = "eval_script.py"
INTER_TEST_PATH = "lfm-challenge.inter_test"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_ndcg_from_stdout(stdout: str) -> Optional[float]:
    """
    Extract the nDCG@10 value from stdout of the evaluation script.

    Args:
        stdout (str): The standard output captured from evaluator.

    Returns:
        Optional[float]: The parsed nDCG@10 score, or None if parsing fails.
    """
    for line in stdout.splitlines():
        if "nDCG@10" in line:
            try:
                return float(line.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                continue

    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?", stdout)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None

    return None


def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna objective function.

    Samples and normalizes weights, builds a submission,
    runs evaluator, and returns nDCG@10.

    Args:
        trial (optuna.trial.Trial): Current optimization trial.

    Returns:
        float: The resulting nDCG@10 score (maximize).
    """
    w_knn = trial.suggest_float("w_knn", 0.0, 1.0)
    w_svd = trial.suggest_float("w_svd", 0.0, 1.0)
    w_pop = trial.suggest_float("w_pop", 0.0, 1.0)

    total = w_knn + w_svd + w_pop
    if total == 0.0:
        w_knn, w_svd, w_pop = 1 / 3, 1 / 3, 1 / 3
    else:
        w_knn /= total
        w_svd /= total
        w_pop /= total

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as tmp_file:
            tmp_filename = tmp_file.name

        build_and_write_submission(
            W_KNN=w_knn,
            W_SVD=w_svd,
            W_POP=w_pop,
            out_filename=tmp_filename,
            top_k=10
        )

        cmd = [
            sys.executable,
            EVAL_SCRIPT,
            "--submission", tmp_filename,
            "--target", INTER_TEST_PATH,
            "--topK", "10"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"[Trial {trial.number}] Evaluation failed with return code {result.returncode}")
            logger.debug(result.stdout)
            logger.debug(result.stderr)
            return 0.0

        ndcg = parse_ndcg_from_stdout(result.stdout)
        if ndcg is None:
            logger.warning(f"[Trial {trial.number}] Failed to parse nDCG@10.")
            return 0.0

        logger.info(f"[Trial {trial.number}] Weights = "
                    f"(KNN: {w_knn:.3f}, SVD: {w_svd:.3f}, POP: {w_pop:.3f}) → nDCG@10 = {ndcg:.6f}")
        return ndcg

    except Exception as e:
        logger.exception(f"[Trial {trial.number}] Exception occurred: {e}")
        return 0.0

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


def normalize_weights(params: dict) -> tuple[float, float, float]:
    """
    Normalize the weight parameters to ensure they sum to 1.

    Args:
        params (dict): Dictionary with raw weights for w_knn, w_svd, w_pop.

    Returns:
        Tuple of normalized weights (w_knn, w_svd, w_pop).
    """
    w_knn, w_svd, w_pop = params["w_knn"], params["w_svd"], params["w_pop"]
    total = w_knn + w_svd + w_pop
    if total == 0.0:
        return 1 / 3, 1 / 3, 1 / 3
    return w_knn / total, w_svd / total, w_pop / total


def main() -> None:
    """Main execution entry point."""
    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_ndcg = study.best_value
    w_knn, w_svd, w_pop = normalize_weights(best_params)

    logger.info("\n=== Optuna Tuning Complete ===")
    logger.info(f"Best raw parameters: {best_params}")
    logger.info(f"Normalized weights: W_KNN={w_knn:.4f}, W_SVD={w_svd:.4f}, W_POP={w_pop:.4f}")
    logger.info(f"Best nDCG@10: {best_ndcg:.6f}")

    if input("\nWrite a final submission TSV with these best weights? (y/n) ").strip().lower() == "y":
        final_file = "optuna_final_submission.tsv"
        build_and_write_submission(
            W_KNN=w_knn,
            W_SVD=w_svd,
            W_POP=w_pop,
            out_filename=final_file,
            top_k=10
        )
        logger.info(f"Final submission written to {final_file}")
        logger.info(f"To evaluate:\n  {sys.executable} {EVAL_SCRIPT} "
                    f"--submission {final_file} --target {INTER_TEST_PATH} --topK 10")


if __name__ == "__main__":
    main()

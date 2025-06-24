#!/usr/bin/env python3
"""
bayes_tune.py

Standalone Optuna tuning script for (W_KNN, W_SVD, W_POP) by maximizing nDCG@10.
No external dependencies on build_submission.py.
"""

import os
import sys
import re
import subprocess
import optuna
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# Path to the evaluation script and test interactions
EVAL_SCRIPT = "eval_script.py"
INTER_TEST_PATH = "lfm-challenge.inter_test"


def load_tsv(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")
    return pd.read_csv(path, sep="\t")


def build_and_write_submission(
    user_path="lfm-challenge.user",
    item_path="lfm-challenge.item",
    inter_train_path="lfm-challenge.inter_train",
    inter_test_path="lfm-challenge.inter_test",
    musicnn_path="lfm-challenge.musicnn",
    out_filename="submission.tsv",
    W_KNN=1.0,
    W_SVD=0.5,
    W_POP=0.2,
    top_k=10,
):
    print("🔄  Loading data …")
    user_df = load_tsv(user_path, "user")
    item_df = load_tsv(item_path, "item")
    inter_train = load_tsv(inter_train_path, "inter_train")
    inter_test = load_tsv(inter_test_path, "inter_test")
    musicnn_df = load_tsv(musicnn_path, "musicnn")

    test_users = inter_test["user_id"].unique()
    N_USERS = int(max(inter_train["user_id"].max(), inter_test["user_id"].max()) + 1)
    N_ITEMS = int(item_df["item_id"].max() + 1)
    VALID_ITEMS = set(item_df["item_id"].unique())

    user_interactions = inter_train.groupby("user_id")["item_id"].apply(set).to_dict()

    def keep_valid(seq, seen=frozenset(), k=top_k):
        out = []
        for itm in seq:
            if itm in VALID_ITEMS and itm not in seen:
                out.append(itm)
                if len(out) == k:
                    break
        return out

    # POP
    popularity = inter_train.groupby("item_id")["user_id"].size().sort_values(ascending=False)
    TOP_POP = [itm for itm in popularity.index if itm in VALID_ITEMS]

    def recommend_pop(user_id, top_n=top_k):
        seen = user_interactions.get(user_id, set())
        return keep_valid(TOP_POP, seen, top_n)

    # ItemKNN
    print("🔄  Building ItemKNN similarity matrix …")
    item_embeddings = musicnn_df.set_index("item_id").sort_index()
    embedding_mat = item_embeddings.values
    item_ids = item_embeddings.index.tolist()
    sim_mat = cosine_similarity(embedding_mat)
    K_NEIGHBORS = 50

    item_knn = {}
    for idx, itm in enumerate(item_ids):
        neigh_idxs = np.argpartition(-sim_mat[idx], range(1, K_NEIGHBORS + 1))[1 : K_NEIGHBORS + 1]
        neighbors = [item_ids[i] for i in neigh_idxs if item_ids[i] in VALID_ITEMS]
        item_knn[itm] = neighbors

    def recommend_itemknn(user_id, top_n=top_k):
        seen = user_interactions.get(user_id, set())
        scores = defaultdict(float)
        for itm in seen:
            for rank, sim_itm in enumerate(item_knn.get(itm, []), start=1):
                if sim_itm not in seen:
                    scores[sim_itm] += (K_NEIGHBORS + 1 - rank)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return keep_valid(ranked, seen, top_n)

    # SVD
    print("🔄  Training SVD …")
    svd_mat = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
    np.add.at(svd_mat, (inter_train["user_id"], inter_train["item_id"]), 1.0)

    svd = TruncatedSVD(n_components=64, n_iter=10, random_state=42)
    user_factors = svd.fit_transform(svd_mat)
    item_factors = svd.components_.T
    svd_scores = user_factors @ item_factors.T

    def recommend_svd(user_id, top_n=top_k):
        seen = user_interactions.get(user_id, set())
        scores = svd_scores[user_id]
        cand = np.argpartition(-scores, range(top_n * 5))[: top_n * 5]
        ranked = cand[np.argsort(-scores[cand])]
        return keep_valid(ranked, seen, top_n)

    print("🔄  Generating hybrid recommendations …")
    hybrid_recs = {}
    for u in tqdm(test_users, desc="▶ Users"):
        scores = defaultdict(float)
        for rank, itm in enumerate(recommend_itemknn(u, 50), start=1):
            scores[itm] += W_KNN * (51 - rank)
        for rank, itm in enumerate(recommend_svd(u, 50), start=1):
            scores[itm] += W_SVD * (51 - rank)
        for rank, itm in enumerate(recommend_pop(u, 50), start=1):
            scores[itm] += W_POP * (51 - rank)

        ranked_items = sorted(scores, key=scores.get, reverse=True)
        hybrid_recs[u] = keep_valid(ranked_items, user_interactions.get(u, set()), top_k)

    bad_ids = {i for rec in hybrid_recs.values() for i in rec} - VALID_ITEMS
    if bad_ids:
        raise AssertionError(f"Out-of-range item_ids detected: {sorted(bad_ids)[:5]}")

    print(f"💾  Writing submission file → {out_filename}")
    with open(out_filename, "w") as f:
        for u, rec in sorted(hybrid_recs.items()):
            if rec:
                f.write(f"{u}\t{','.join(map(str, rec))}\n")
    return out_filename


def parse_ndcg_from_stdout(stdout: str) -> float:
    for line in stdout.splitlines():
        if "nDCG@10" in line:
            try:
                return float(line.split(":", 1)[1].strip())
            except Exception:
                pass
    float_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?")
    all_floats = float_pattern.findall(stdout)
    if all_floats:
        try:
            return float(all_floats[-1])
        except Exception:
            return None
    return None


def objective(trial: optuna.trial.Trial) -> float:
    w_knn = trial.suggest_float("w_knn", 0.0, 1.0)
    w_svd = trial.suggest_float("w_svd", 0.0, 1.0)
    w_pop = trial.suggest_float("w_pop", 0.0, 1.0)
    total = w_knn + w_svd + w_pop
    if total == 0.0:
        w_knn, w_svd, w_pop = 1/3, 1/3, 1/3
    else:
        w_knn /= total
        w_svd /= total
        w_pop /= total

    tmp_filename = f"optuna_tmp_{trial.number}.tsv"
    try:
        build_and_write_submission(W_KNN=w_knn, W_SVD=w_svd, W_POP=w_pop, out_filename=tmp_filename, top_k=10)
    except Exception as e:
        print(f"[Trial {trial.number}] build failed:", e)
        return 0.0

    cmd = [sys.executable, EVAL_SCRIPT, "--submission", tmp_filename, "--target", INTER_TEST_PATH, "--topK", "10"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[Trial {trial.number}] eval failed", proc.stdout, proc.stderr)
        os.remove(tmp_filename)
        return 0.0

    ndcg = parse_ndcg_from_stdout(proc.stdout)
    os.remove(tmp_filename)
    if ndcg is None:
        print(f"[Trial {trial.number}] Failed to parse nDCG", proc.stdout)
        return 0.0

    print(f"[Trial {trial.number}] Weights = ({w_knn:.3f}, {w_svd:.3f}, {w_pop:.3f}) → nDCG@10 = {ndcg:.6f}")
    return ndcg


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\n=== Optuna Tuning Complete ===")
    print("Best raw parameters:", study.best_params)

    w_knn_b = study.best_params["w_knn"]
    w_svd_b = study.best_params["w_svd"]
    w_pop_b = study.best_params["w_pop"]
    total_b = w_knn_b + w_svd_b + w_pop_b
    if total_b == 0.0:
        w_knn_b, w_svd_b, w_pop_b = 1/3, 1/3, 1/3
    else:
        w_knn_b /= total_b
        w_svd_b /= total_b
        w_pop_b /= total_b

    print(f"Normalized best weights → W_KNN = {w_knn_b:.4f}, W_SVD = {w_svd_b:.4f}, W_POP = {w_pop_b:.4f}")
    print("Best nDCG@10 =", study.best_value)

    answer = input("\nWrite a final submission TSV with these best weights? (y/n) ").strip().lower()
    if answer == "y":
        final_file = "optuna_final_submission.tsv"
        build_and_write_submission(W_KNN=w_knn_b, W_SVD=w_svd_b, W_POP=w_pop_b, out_filename=final_file, top_k=10)
        print(f"Final TSV written to {final_file}. To evaluate it, run:")
        print(f"  {sys.executable} {EVAL_SCRIPT} --submission {final_file} --target {INTER_TEST_PATH} --topK 10")

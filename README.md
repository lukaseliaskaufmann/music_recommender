# 🎵 Music Recommender System – Weight Optimization with Optuna

**Author:** Lukas Kaufmann
**Project Type:** Learning from user generated data - Challenge Submission 
**Goal:** Optimize a hybrid music recommender system using ensemble learning and Bayesian optimization.

---

## 📌 Overview

This project implements a **music recommender system** that combines multiple models using weighted ensembling. The core objective is to recommend personalized music tracks to users and maximize the **nDCG@10** ranking metric.

The final model combines:

- 🎯 **ItemKNN** (Item-based Collaborative Filtering)
- 🔍 **SVD** (Matrix Factorization)
- 🌍 **Popularity Model** (Global item frequency baseline)

Weights are automatically tuned using **Bayesian optimization via Optuna**, achieving strong results on the challenge platform (Codalab).

---

## 🚀 Key Features

- **Hybrid Recommendation System**: Combines collaborative and non-personalized methods.
- **Optuna Integration**: Automatically learns optimal model weights.
- **Reproducible Pipeline**: Scripts for tuning, submission generation, and evaluation.
- **Challenge Compatible**: Designed for the [lfm-challenge] dataset format.

---

## 📊 Final Results

| System                      | nDCG@10                                              |
| --------------------------- | ---------------------------------------------------- |
| ItemKNN                     | ~0.107                                               |
| SVD                         | ~0.129                                               |
| Popularity                  | ~0.027                                               |
| Equal-weight Ensemble       | ~0.112                                               |
| 🎯**Optuna Ensemble** | **0.1322 (local)** / **0.195 (Codalab)** |

---

## 🛠 Repository Structure

```
├── bayes_tune.py          # Optuna-based weight optimizer
├── build_submission.py    # Generates hybrid recommendation output
├── eval_script.py         # Evaluates submission using nDCG@10
├── optuna_final_submission.tsv # Final submission file (sample)
├── README.md              # This file
```

---

## 🧪 How It Works

1. Each model computes a recommendation score per user-item pair.
2. The scores are combined using a weighted average:
   ```
   final_score = w_knn * KNN_score + w_svd * SVD_score + w_pop * Popularity_score
   ```
3. Optuna explores different weight combinations to maximize validation nDCG@10.
4. The best combination is used to generate a final submission.

import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss
)

ACTION_FEATURES = [
    "agent_role", "tool_name", "action_type", "data_sensitivity",
    "destination_category", "timestamp_norm", "task_type", "agent_count",
]
TRAJECTORY_FEATURES = ACTION_FEATURES + [
    "mean_sensitivity", "max_sensitivity", "mean_destination",
    "max_destination", "prev_tool_count", "prev_sensitive_reads",
    "prev_external_transmits", "prev_risky_actions", "max_prev_risk",
    "cumulative_risk",
]


def evaluate_on_csv(model, features, csv_path):
    df = pd.read_csv(csv_path)
    X = df[features]
    y = df["label"]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auroc": roc_auc_score(y, y_prob),
        "auprc": average_precision_score(y, y_prob),
        "brier": brier_score_loss(y, y_prob),
    }


def find_optimal_thresholds(traj_model, train_csv="data/trajectories.csv",
                             seed=99):
    df = pd.read_csv(train_csv)
    val_df = df.sample(frac=0.2, random_state=seed)
    X_val = val_df[TRAJECTORY_FEATURES]
    y_val = val_df["label"]
    y_prob = traj_model.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_thresholds = (0.25, 0.50, 0.75)
    for t1 in np.arange(0.15, 0.40, 0.05):
        for t2 in np.arange(0.40, 0.65, 0.05):
            for t3 in np.arange(0.65, 0.85, 0.05):
                y_pred = (y_prob >= t1).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                fbr = ((y_prob >= t1) & (y_val == 0)).sum() / max(
                    (y_val == 0).sum(), 1)
                if f1 > best_f1 and fbr <= 0.05:
                    best_f1 = f1
                    best_thresholds = (t1, t2, t3)

    return {
        "autonomous_max": float(best_thresholds[0]),
        "restricted_max": float(best_thresholds[1]),
        "approval_max": float(best_thresholds[2]),
    }, best_f1


if __name__ == "__main__":
    import pickle
    with open("models/action_level_model.pkl", "rb") as f:
        action_model = pickle.load(f)
    with open("models/trajectory_level_model.pkl", "rb") as f:
        traj_model = pickle.load(f)

    print("HOLDOUT:")
    print(" action:", evaluate_on_csv(action_model, ACTION_FEATURES,
                                       "data/trajectories_holdout.csv"))
    print(" traj:  ", evaluate_on_csv(traj_model, TRAJECTORY_FEATURES,
                                       "data/trajectories_holdout.csv"))

    print("COMPOSITIONAL:")
    print(" action:", evaluate_on_csv(action_model, ACTION_FEATURES,
                                       "data/trajectories_compositional.csv"))
    print(" traj:  ", evaluate_on_csv(traj_model, TRAJECTORY_FEATURES,
                                       "data/trajectories_compositional.csv"))

    thresholds, f1 = find_optimal_thresholds(traj_model)
    print("thresholds:", thresholds, "f1:", f1)
    with open("models/thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
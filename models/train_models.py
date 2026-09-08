import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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
LABEL_COL = "label"


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "brier": brier_score_loss(y_test, y_prob),
    }


def train_model(df, features, n_estimators=200, max_depth=5,
                 learning_rate=0.1, seed=42):
    X = df[features]
    y = df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=seed
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics


def train_both(csv_path="data/trajectories.csv", suffix="", seed=42,
                save=True):
    df = pd.read_csv(csv_path)
    action_model, action_metrics = train_model(df, ACTION_FEATURES, seed=seed)
    traj_model, traj_metrics = train_model(df, TRAJECTORY_FEATURES, seed=seed)

    if save:
        with open(f"models/action_level_model{suffix}.pkl", "wb") as f:
            pickle.dump(action_model, f)
        with open(f"models/trajectory_level_model{suffix}.pkl", "wb") as f:
            pickle.dump(traj_model, f)

    return action_model, traj_model, action_metrics, traj_metrics


if __name__ == "__main__":
    a_model, t_model, a_metrics, t_metrics = train_both()
    print("Action-level:", a_metrics)
    print("Trajectory-level:", t_metrics)
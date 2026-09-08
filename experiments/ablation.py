"""
Ablation over trajectory features. Compares four feature-set variants:
  A) action-only              (the baseline action-level model)
  B) action + history length  (just "how many steps happened before")
  C) action + trajectory stats, excluding risk-derived features
  D) full trajectory feature set (the trajectory-level model)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from models.train_models import evaluate_model

BASE = [
    "agent_role", "tool_name", "action_type", "data_sensitivity",
    "destination_category", "timestamp_norm", "task_type", "agent_count",
]

FEATURE_SETS = {
    "A_action_only": BASE,
    "B_action_plus_history_length": BASE + ["prev_tool_count"],
    "C_action_plus_traj_stats_no_risk": BASE + [
        "mean_sensitivity", "max_sensitivity", "mean_destination",
        "max_destination", "prev_tool_count", "prev_sensitive_reads",
        "prev_external_transmits",
    ],
    "D_full_trajectory": BASE + [
        "mean_sensitivity", "max_sensitivity", "mean_destination",
        "max_destination", "prev_tool_count", "prev_sensitive_reads",
        "prev_external_transmits", "prev_risky_actions", "max_prev_risk",
        "cumulative_risk",
    ],
}


def train_and_eval(df_train, features, df_test, seed=42):
    X_train = df_train[features]
    y_train = df_train["label"]
    model = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                        learning_rate=0.1, random_state=seed)
    model.fit(X_train, y_train)
    X_test = df_test[features]
    y_test = df_test["label"]
    return evaluate_model(model, X_test, y_test)


def main(train_csv="data/trajectories.csv",
         holdout_csv="data/trajectories_holdout.csv",
         comp_csv="data/trajectories_compositional.csv"):
    df_train_full = pd.read_csv(train_csv)
    df_holdout = pd.read_csv(holdout_csv)
    df_comp = pd.read_csv(comp_csv)

    df_train, df_internal_test = train_test_split(
        df_train_full, test_size=0.2, random_state=42,
        stratify=df_train_full["label"]
    )

    rows = []
    for name, features in FEATURE_SETS.items():
        m_internal = train_and_eval(df_train, features, df_internal_test)
        m_holdout = train_and_eval(df_train, features, df_holdout)
        m_comp = train_and_eval(df_train, features, df_comp)
        rows.append({
            "feature_set": name,
            "n_features": len(features),
            "internal_f1": m_internal["f1"],
            "internal_brier": m_internal["brier"],
            "holdout_f1": m_holdout["f1"],
            "holdout_brier": m_holdout["brier"],
            "comp_f1": m_comp["f1"],
            "comp_brier": m_comp["brier"],
        })
        print(f"{name}: holdout_f1={m_holdout['f1']:.3f} "
              f"comp_f1={m_comp['f1']:.3f} comp_brier={m_comp['brier']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/ablation.csv", index=False)
    print("\nSaved to experiments/results/ablation.csv")
    return df


if __name__ == "__main__":
    main()
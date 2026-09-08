"""
Runs the full pipeline (data gen -> train -> evaluate -> governance
benchmark) across multiple random seeds, then reports mean + 95% bootstrap
CI for every headline metric.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
import json

from data.generate_trajectories import (
    generate_dataset, generate_compositional_dataset,
    ACTION_FEATURES, TRAJECTORY_FEATURES
)
from models.train_models import train_both, evaluate_model
from models.evaluate_models import evaluate_on_csv, find_optimal_thresholds
from experiments.run_experiments import (
    run_governance_benchmark, CONVENTIONAL_ATTACKS, PATH_DEPENDENT_ATTACKS
)

SEEDS = [1, 2, 3, 4, 5]
N_BOOTSTRAP = 2000


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=95):
    values = np.array(values)
    boot_means = [np.mean(np.random.choice(values, size=len(values),
                                            replace=True))
                  for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(np.mean(values)), float(lower), float(upper)


def evaluate_on_csv_pkl(model_type, suffix, csv_path):
    import pickle
    path = f"models/{'action_level' if model_type=='action' else 'trajectory_level'}_model{suffix}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    features = ACTION_FEATURES if model_type == "action" else TRAJECTORY_FEATURES
    return evaluate_on_csv(model, features, csv_path)


def run_one_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    suffix = f"_seed{seed}"
    generate_dataset(suffix=suffix)
    generate_compositional_dataset(suffix=suffix)

    _, _, action_metrics, traj_metrics = train_both(
        csv_path=f"data/trajectories{suffix}.csv", suffix=suffix, seed=seed
    )

    holdout_action = evaluate_on_csv_pkl("action", suffix,
                                          f"data/trajectories_holdout{suffix}.csv")
    holdout_traj = evaluate_on_csv_pkl("trajectory", suffix,
                                        f"data/trajectories_holdout{suffix}.csv")
    comp_action = evaluate_on_csv_pkl("action", suffix,
                                       f"data/trajectories_compositional{suffix}.csv")
    comp_traj = evaluate_on_csv_pkl("trajectory", suffix,
                                     f"data/trajectories_compositional{suffix}.csv")

    import pickle
    with open(f"models/trajectory_level_model{suffix}.pkl", "rb") as f:
        traj_model = pickle.load(f)
    thresholds, _ = find_optimal_thresholds(
        traj_model, train_csv=f"data/trajectories{suffix}.csv", seed=seed + 100
    )

    model_paths = {
        "action": f"models/action_level_model{suffix}.pkl",
        "trajectory": f"models/trajectory_level_model{suffix}.pkl",
    }

    gov_results = {}
    for gtype in ["none", "static", "action", "trajectory"]:
        r = run_governance_benchmark(
            gtype, PATH_DEPENDENT_ATTACKS, n_benign=30, n_attack=60,
            thresholds=thresholds, seed=seed + 1000, model_paths=model_paths
        )
        gov_results[gtype] = r

    return {
        "seed": seed,
        "holdout_action_f1": holdout_action["f1"],
        "holdout_traj_f1": holdout_traj["f1"],
        "holdout_action_brier": holdout_action["brier"],
        "holdout_traj_brier": holdout_traj["brier"],
        "comp_action_f1": comp_action["f1"],
        "comp_traj_f1": comp_traj["f1"],
        "comp_action_brier": comp_action["brier"],
        "comp_traj_brier": comp_traj["brier"],
        "gov_static_attack_success": gov_results["static"]["attack_success_rate"],
        "gov_action_attack_success": gov_results["action"]["attack_success_rate"],
        "gov_traj_attack_success": gov_results["trajectory"]["attack_success_rate"],
        "gov_action_false_block": gov_results["action"]["false_block_rate"],
        "gov_traj_false_block": gov_results["trajectory"]["false_block_rate"],
    }


def main():
    all_runs = []
    for seed in SEEDS:
        print(f"=== Seed {seed} ===")
        result = run_one_seed(seed)
        print(result)
        all_runs.append(result)

    df = pd.DataFrame(all_runs)
    df.to_csv("experiments/results/multi_seed_raw.csv", index=False)

    summary = {}
    for col in df.columns:
        if col == "seed":
            continue
        mean, lo, hi = bootstrap_ci(df[col].values)
        summary[col] = {"mean": mean, "ci_lower": lo, "ci_upper": hi}

    with open("experiments/results/multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY (mean [95% CI]) ===")
    for k, v in summary.items():
        print(f"{k:35s}: {v['mean']:.4f}  [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")

    return df, summary


if __name__ == "__main__":
    main()
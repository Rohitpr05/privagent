"""
Generates all figures needed for the paper from the Step A result files.
Saves PNGs to plots/ at 300 DPI (LaTeX-ready).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def fig_f1_brier_comparison():
    with open("experiments/results/multi_seed_summary.json") as f:
        s = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    labels = ["Held-out\n(unseen attacks)", "Path-dependent\n(compositional)"]
    action_vals = [s["holdout_action_f1"]["mean"], s["comp_action_f1"]["mean"]]
    traj_vals = [s["holdout_traj_f1"]["mean"], s["comp_traj_f1"]["mean"]]
    action_err = [[s["holdout_action_f1"]["mean"] - s["holdout_action_f1"]["ci_lower"],
                   s["comp_action_f1"]["mean"] - s["comp_action_f1"]["ci_lower"]],
                  [s["holdout_action_f1"]["ci_upper"] - s["holdout_action_f1"]["mean"],
                   s["comp_action_f1"]["ci_upper"] - s["comp_action_f1"]["mean"]]]
    traj_err = [[s["holdout_traj_f1"]["mean"] - s["holdout_traj_f1"]["ci_lower"],
                 s["comp_traj_f1"]["mean"] - s["comp_traj_f1"]["ci_lower"]],
                [s["holdout_traj_f1"]["ci_upper"] - s["holdout_traj_f1"]["mean"],
                 s["comp_traj_f1"]["ci_upper"] - s["comp_traj_f1"]["mean"]]]

    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w/2, action_vals, w, yerr=action_err, capsize=4,
                label="Action-level", color="#a8b3c5")
    axes[0].bar(x + w/2, traj_vals, w, yerr=traj_err, capsize=4,
                label="Trajectory-level", color="#2e5f8a")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("F1 score"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) F1 score, 95% bootstrap CI")
    axes[0].legend(frameon=False, loc="lower right")

    action_vals_b = [s["holdout_action_brier"]["mean"], s["comp_action_brier"]["mean"]]
    traj_vals_b = [s["holdout_traj_brier"]["mean"], s["comp_traj_brier"]["mean"]]
    action_err_b = [[s["holdout_action_brier"]["mean"] - s["holdout_action_brier"]["ci_lower"],
                     s["comp_action_brier"]["mean"] - s["comp_action_brier"]["ci_lower"]],
                    [s["holdout_action_brier"]["ci_upper"] - s["holdout_action_brier"]["mean"],
                     s["comp_action_brier"]["ci_upper"] - s["comp_action_brier"]["mean"]]]
    traj_err_b = [[s["holdout_traj_brier"]["mean"] - s["holdout_traj_brier"]["ci_lower"],
                   s["comp_traj_brier"]["mean"] - s["comp_traj_brier"]["ci_lower"]],
                  [s["holdout_traj_brier"]["ci_upper"] - s["holdout_traj_brier"]["mean"],
                   s["comp_traj_brier"]["ci_upper"] - s["comp_traj_brier"]["mean"]]]

    axes[1].bar(x - w/2, action_vals_b, w, yerr=action_err_b, capsize=4,
                label="Action-level", color="#a8b3c5")
    axes[1].bar(x + w/2, traj_vals_b, w, yerr=traj_err_b, capsize=4,
                label="Trajectory-level", color="#2e5f8a")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Brier score (lower is better)")
    axes[1].set_title("(b) Calibration, 95% bootstrap CI")
    axes[1].legend(frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/fig1_f1_brier_comparison.png")
    plt.close()
    print("Saved plots/fig1_f1_brier_comparison.png")


def fig_governance_comparison():
    with open("experiments/results/multi_seed_summary.json") as f:
        s = json.load(f)

    strategies = ["Static\nauthorization", "Action-level\nrisk", "Trajectory-level\nrisk"]
    means = [s["gov_static_attack_success"]["mean"],
             s["gov_action_attack_success"]["mean"],
             s["gov_traj_attack_success"]["mean"]]
    lo = [s["gov_static_attack_success"]["ci_lower"],
          s["gov_action_attack_success"]["ci_lower"],
          s["gov_traj_attack_success"]["ci_lower"]]
    hi = [s["gov_static_attack_success"]["ci_upper"],
          s["gov_action_attack_success"]["ci_upper"],
          s["gov_traj_attack_success"]["ci_upper"]]
    err = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = ["#c0392b", "#e0a458", "#2e5f8a"]
    ax.bar(strategies, means, yerr=err, capsize=6, color=colors)
    ax.set_ylabel("Attack success rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Path-dependent attacks: static rules fail completely\n"
                 "(mean \u00b1 95% bootstrap CI, 5 seeds)")
    for i, m in enumerate(means):
        ax.text(i, m + 0.03, f"{m:.2f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/fig2_governance_attack_success.png")
    plt.close()
    print("Saved plots/fig2_governance_attack_success.png")


def fig_threshold_sensitivity():
    df = pd.read_csv("experiments/results/threshold_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for model_type, color, marker in [("action", "#a8b3c5", "o"),
                                       ("trajectory", "#2e5f8a", "s")]:
        sub = df[df["model_type"] == model_type].sort_values("threshold")
        label = "Action-level" if model_type == "action" else "Trajectory-level"
        ax.plot(sub["threshold"], sub["attack_success_rate"], marker=marker,
                color=color, label=label, linewidth=2, markersize=5)
    ax.axvspan(0.15, 0.40, alpha=0.08, color="green",
               label="Chosen operating\nrange (0.15\u20130.40)")
    ax.set_xlabel("Block threshold"); ax.set_ylabel("Attack success rate")
    ax.set_title("Threshold sensitivity: attack success vs. block threshold")
    ax.legend(frameon=False); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig("plots/fig3_threshold_sensitivity.png")
    plt.close()
    print("Saved plots/fig3_threshold_sensitivity.png")


def fig_ablation():
    df = pd.read_csv("experiments/results/ablation.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(df)); w = 0.35
    ax.bar(x - w/2, df["holdout_f1"], w, label="Held-out F1", color="#a8b3c5")
    ax.bar(x + w/2, df["comp_f1"], w, label="Path-dependent F1", color="#2e5f8a")
    ax.set_xticks(x)
    labels = ["A: action\nonly", "B: +history\nlength", "C: +traj stats\n(no risk)",
              "D: full\ntrajectory"]
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.0)
    ax.set_title("Ablation: which trajectory features drive the improvement?")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("plots/fig4_ablation.png")
    plt.close()
    print("Saved plots/fig4_ablation.png")


if __name__ == "__main__":
    fig_f1_brier_comparison()
    fig_governance_comparison()
    fig_threshold_sensitivity()
    fig_ablation()
    print("\nAll figures saved to plots/")
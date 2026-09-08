import pickle
import json
import numpy as np
import pandas as pd

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


class RiskEngine:
    def __init__(self, model_type="trajectory", model_path=None,
                 thresholds_path="models/thresholds.json"):
        self.model_type = model_type
        if model_type == "trajectory":
            path = model_path or "models/trajectory_level_model.pkl"
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.features = TRAJECTORY_FEATURES
        elif model_type == "action":
            path = model_path or "models/action_level_model.pkl"
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.features = ACTION_FEATURES
        else:
            self.model = None
            self.features = None

        with open(thresholds_path, "r") as f:
            self.thresholds = json.load(f)

    def compute_risk(self, trajectory_prefix, candidate_action,
                      task_type, agent_count):
        if self.model_type == "none":
            return 0.0
        if self.model_type == "static":
            if (candidate_action["data_sensitivity"] == 1 and
                    candidate_action["destination_category"] == 2):
                return 0.9
            return 0.1

        action_feats = {
            "agent_role": candidate_action["agent_role"],
            "tool_name": candidate_action["tool_name"],
            "action_type": candidate_action["action_type"],
            "data_sensitivity": candidate_action["data_sensitivity"],
            "destination_category": candidate_action["destination_category"],
            "timestamp_norm": candidate_action.get("timestamp_norm", 0.5),
            "task_type": task_type,
            "agent_count": agent_count,
        }

        if self.model_type == "action":
            feature_vector = [action_feats[f] for f in ACTION_FEATURES]
        else:
            if len(trajectory_prefix) == 0:
                traj_feats = {
                    "mean_sensitivity": 0.0, "max_sensitivity": 0.0,
                    "mean_destination": 0.0, "max_destination": 0.0,
                    "prev_tool_count": 0, "prev_sensitive_reads": 0,
                    "prev_external_transmits": 0, "prev_risky_actions": 0,
                    "max_prev_risk": 0.0, "cumulative_risk": 0.0,
                }
            else:
                sensitivities = [a["data_sensitivity"]
                                  for a in trajectory_prefix]
                destinations = [a["destination_category"]
                                 for a in trajectory_prefix]
                ext_transmits = sum(1 for a in trajectory_prefix
                                     if a["destination_category"] == 2)
                sens_reads = sum(1 for a in trajectory_prefix
                                  if a["data_sensitivity"] == 1)
                traj_feats = {
                    "mean_sensitivity": np.mean(sensitivities),
                    "max_sensitivity": float(max(sensitivities)),
                    "mean_destination": np.mean(destinations),
                    "max_destination": float(max(destinations)),
                    "prev_tool_count": len(trajectory_prefix),
                    "prev_sensitive_reads": sens_reads,
                    "prev_external_transmits": ext_transmits,
                    "prev_risky_actions": sens_reads + ext_transmits,
                    "max_prev_risk": float(max(destinations)) / 2.0,
                    "cumulative_risk": float(sens_reads + ext_transmits)
                                        / max(len(trajectory_prefix), 1),
                }
            all_feats = {**action_feats, **traj_feats}
            feature_vector = [all_feats[f] for f in TRAJECTORY_FEATURES]

        cols = ACTION_FEATURES if self.model_type == "action" else TRAJECTORY_FEATURES
        X = pd.DataFrame([feature_vector], columns=cols)
        risk_score = self.model.predict_proba(X)[0][1]
        return float(risk_score)


class AutonomyController:
    def __init__(self, thresholds_path="models/thresholds.json",
                 thresholds=None):
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)

    def decide(self, risk_score):
        t1 = self.thresholds["autonomous_max"]
        t2 = self.thresholds["restricted_max"]
        t3 = self.thresholds["approval_max"]
        if risk_score < t1:
            return "AUTONOMOUS"
        elif risk_score < t2:
            return "RESTRICTED"
        elif risk_score < t3:
            return "HUMAN_APPROVAL"
        else:
            return "BLOCKED"
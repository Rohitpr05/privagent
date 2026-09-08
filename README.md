# privagent

run this 

# Part 1: base pipeline
python data/generate_trajectories.py

python models/train_models.py

python models/evaluate_models.py

python experiments/run_experiments.py

# Part 2: Step A strengthening (takes a few minutes total)
python experiments/multi_seed_run.py

python experiments/threshold_sensitivity.py

python experiments/ablation.py

python experiments/make_plots.py

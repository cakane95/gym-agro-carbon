import os
import json
from pathlib import Path
import numpy as np

def setup_experiment_dirs(exp_name: str):
    """
    Crée l'arborescence des dossiers pour une expérience donnée.
    """
    base_path = Path("results")
    dirs = {
        "exp": base_path / "experiments" / exp_name,
        "logs": base_path / "logs" / exp_name,
        "summary": base_path / "summary" / exp_name
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    return dirs

def save_summary_stats(exp_dirs, cumulative_regrets: list):
    """
    Calcule et sauvegarde les statistiques descriptives (Q1, Q3, moyenne, etc.).
    """
    stats = {
        "mean_regret": np.mean(cumulative_regrets),
        "std_regret": np.std(cumulative_regrets),
        "median_regret": np.median(cumulative_regrets),
        "q1_regret": np.percentile(cumulative_regrets, 25),
        "q3_regret": np.percentile(cumulative_regrets, 75),
    }
    
    summary_path = exp_dirs["summary"] / "stats.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"✅ Stats saved to {summary_path}")
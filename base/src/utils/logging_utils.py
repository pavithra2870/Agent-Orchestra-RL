"""Episode-level logging and plotting for Phase 1 baselines."""

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class EpisodeLogger:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def log(self, **kwargs) -> None:
        self.records.append(kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


def summarize(df: pd.DataFrame, group_col: str = "agent") -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .agg(
            episodes=("reward", "count"),
            avg_reward=("reward", "mean"),
            success_rate=("success", "mean"),
            avg_cost=("cost", "mean"),
            avg_latency=("latency", "mean"),
            avg_steps=("steps", "mean"),
        )
        .reset_index()
    )


def plot_learning_curve(df: pd.DataFrame, path: str, window: int = 100, value_col: str = "success") -> None:
    """Rolling-mean learning curve of `value_col` (e.g. success or reward)
    per training episode, one line per agent. Only meaningful for agents
    logged during *training* (agent column should already be filtered)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent_name, sub in df.groupby("agent"):
        sub = sub.sort_values("episode")
        rolling = sub[value_col].rolling(window, min_periods=1).mean()
        ax.plot(sub["episode"].values, rolling.values, label=agent_name)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Rolling mean {value_col} (window={window})")
    ax.set_title("Learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_baseline_comparison(summary_df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metrics = ["success_rate", "avg_reward", "avg_cost", "avg_latency"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    for ax, metric in zip(axes, metrics):
        ax.bar(summary_df["agent"], summary_df[metric], color="steelblue")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Baseline comparison (evaluation episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

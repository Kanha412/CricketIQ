"""
CricketIQ — Train Win Probability Model
=========================================
Trains a calibrated Logistic Regression on 2nd-innings ball-by-ball data
and saves the sklearn Pipeline to models/win_probability_model.joblib.

Run:
    python scripts/train_win_model.py

WHY LOGISTIC REGRESSION:
    - ~120K training samples, 7 features: LR converges perfectly
    - Interpretable coefficients (ball#, wickets, RRR all have clear signs)
    - Fast to train (< 5s) and load (< 1ms)
    - CalibratedClassifierCV: ensures probability is accurate, not just ordinal

WHY CALIBRATION:
    Raw LR probabilities are already well-calibrated (it's the optimal
    log-loss model), but CalibratedClassifierCV with cv=5 cross-validates
    the calibration curve — ensuring 70% really means 70%.

PIPELINE:
    StandardScaler → LogisticRegression → CalibratedClassifierCV

    WHY SCALE: required_rr (range 0–36) vs phase_encoded (0–2) differ by 18×.
    Logistic regression's gradient is sensitive to feature scale;
    StandardScaler brings all features to mean=0, std=1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import click
from rich.console import Console
from rich.table import Table
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.features import load_training_data, build_feature_matrix, FEATURE_COLS, TARGET_COL
from src.ml.win_probability import WinProbabilityModel, MODEL_PATH
from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("scripts.train_win_model")
console = Console()


@click.command()
@click.option("--test-size",  default=0.2, show_default=True, help="Fraction of data for test set")
@click.option("--max-iter",   default=500, show_default=True, help="Max iterations for LogisticRegression")
@click.option("--cv-folds",   default=5,   show_default=True, help="Cross-validation folds for calibration")
@click.option("--random-seed",default=42,  show_default=True, help="Random seed for reproducibility")
def main(test_size: float, max_iter: int, cv_folds: int, random_seed: int) -> None:
    """Train and save the CricketIQ win probability model."""

    console.print("\n[bold]CricketIQ — Win Probability Model Training[/bold]")
    console.print(f"Features: {FEATURE_COLS}")

    # ── Load training data ────────────────────────────────────────────────────
    console.print("\n[cyan]Loading training data from database...[/cyan]")
    df = load_training_data()

    if df.empty:
        console.print("[red]No training data found. Run ingestion first.[/red]")
        sys.exit(1)

    X = build_feature_matrix(df)
    y = df[TARGET_COL].to_numpy()

    console.print(f"  Total samples : [bold]{len(X):,}[/bold]")
    console.print(f"  Positive class: [bold]{y.sum():,}[/bold] ({y.mean()*100:.1f}% batting-team wins)")
    console.print(f"  Features      : {X.shape[1]}")

    # ── Train / test split (time-ordered wouldn't help here — no leak) ───────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    console.print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    console.print("\n[cyan]Building pipeline: StandardScaler → LogisticRegression → CalibratedClassifierCV[/cyan]")

    base_lr = LogisticRegression(
        max_iter=max_iter,
        random_state=random_seed,
        solver="lbfgs",
        class_weight="balanced",   # handles minor class imbalance automatically
    )
    # Calibrate inside a sub-pipeline that includes scaling
    inner_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     base_lr),
    ])
    calibrated = CalibratedClassifierCV(
        estimator=inner_pipeline,
        cv=cv_folds,
        method="isotonic",   # isotonic > sigmoid for larger datasets
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    console.print(f"[cyan]Training (cv={cv_folds}, this takes ~30s)...[/cyan]")
    calibrated.fit(X_train, y_train)
    console.print("[green]Training complete.[/green]")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred      = calibrated.predict(X_test)
    y_prob      = calibrated.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)
    brier       = brier_score_loss(y_test, y_prob)
    logloss     = log_loss(y_test, y_prob)

    table = Table(title="Model Evaluation (held-out test set)", show_lines=True)
    table.add_column("Metric",    style="cyan", justify="left")
    table.add_column("Value",     style="bold green", justify="right")
    table.add_column("Interpretation", style="dim")
    table.add_row("Accuracy",         f"{accuracy:.3f}",
                  ">0.70 is good for this problem")
    table.add_row("ROC-AUC",          f"{auc:.3f}",
                  ">0.80 = model ranks outcomes well")
    table.add_row("Brier Score",      f"{brier:.3f}",
                  "<0.20 = well-calibrated probabilities")
    table.add_row("Log Loss",         f"{logloss:.3f}",
                  "Lower is better; random guess = 0.693")
    console.print(table)

    # ── Sanity checks on known scenarios ─────────────────────────────────────
    console.print("\n[cyan]Sanity checks (expected probabilities):[/cyan]")
    from src.ml.features import build_inference_row
    scenarios = [
        ("Start of chase, target 180",       build_inference_row(0,   0, 120, 180, "powerplay"), "~50%"),
        ("Chase 14 off 6 balls, 0 wickets",  build_inference_row(166, 0,   6,  14,   "death"),  ">85%"),
        ("Chase 60 off 6 balls, 3 wickets",  build_inference_row(120, 3,   6,  60,   "death"),  "<5%"),
        ("Chase 36 off 36 balls, 2 wickets", build_inference_row(144, 2,  36,  36,  "middle"),  "~65%"),
    ]
    for label, X_row, expected in scenarios:
        prob = calibrated.predict_proba(X_row)[0][1]
        color = "green" if prob > 0.3 else "red"
        console.print(
            f"  {label:<45} → [{color}]{prob:.2%}[/{color}] (expected {expected})"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    model = WinProbabilityModel(pipeline=calibrated)
    model.save()
    console.print(f"\n[green]✓ Model saved to {MODEL_PATH}[/green]")
    console.print("[dim]Load in app: WinProbabilityModel.load().predict(...)[/dim]\n")


if __name__ == "__main__":
    main()

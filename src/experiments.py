from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import optuna

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loader import load_dataframe, split_X_y
from src.preprocess import build_feature_pipeline
from src.evaluate import evaluate_binary


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def make_model(model_name: str, trial: optuna.Trial | None = None):
    """Create a classifier. If trial is provided, use tuned params."""
    if model_name == "logreg":
        if trial is None:
            return LogisticRegression(max_iter=2000, class_weight="balanced")
        C = trial.suggest_float("C", 1e-3, 50.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        return LogisticRegression(max_iter=3000, C=C, solver=solver, class_weight="balanced")

    if model_name == "rf":
        if trial is None:
            return RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
        n_estimators = trial.suggest_int("n_estimators", 200, 800)
        max_depth = trial.suggest_int("max_depth", 3, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )

    if model_name == "xgb":
        if trial is None:
            return XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            )
        n_estimators = trial.suggest_int("n_estimators", 200, 800)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 10)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )

    if model_name == "lgbm":
        if trial is None:
            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )
        n_estimators = trial.suggest_int("n_estimators", 200, 1200)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 16, 128)
        max_depth = trial.suggest_int("max_depth", -1, 20)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, use_pca: bool, trial: optuna.Trial | None = None) -> Pipeline:
    features = build_feature_pipeline(use_pca=use_pca)
    clf = make_model(model_name, trial=trial)
    return Pipeline(steps=[
        ("features", features),
        ("clf", clf),
    ])


def run_single(model_name: str, use_pca: bool, use_tuning: bool, X_train, y_train, X_test, y_test):
    best_trial_params = None

    if not use_tuning:
        pipe = build_pipeline(model_name, use_pca, trial=None)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        res = evaluate_binary(y_test, preds)
        return pipe, res, best_trial_params

    # Optuna tuning on a validation split (simple + works well for this class project)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    def objective(trial: optuna.Trial):
        pipe = build_pipeline(model_name, use_pca, trial=trial)
        pipe.fit(X_tr, y_tr)
        val_pred = pipe.predict(X_val)
        return float(evaluate_binary(y_val, val_pred).f1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25, show_progress_bar=False)

    best_trial_params = study.best_params

    # Train final model on full train with best params
    pipe = build_pipeline(model_name, use_pca, trial=study.best_trial)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    res = evaluate_binary(y_test, preds)
    return pipe, res, best_trial_params


def save_artifacts(run_id: str, pipeline: Pipeline, result, meta: dict):
    model_path = MODELS_DIR / f"{run_id}.pkl"
    metrics_path = METRICS_DIR / f"{run_id}.json"

    joblib.dump(pipeline, model_path)

    payload = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "f1": result.f1,
        "confusion_matrix": result.confusion_matrix,
        "report": result.report,
        "meta": meta,
        "model_path": str(model_path),
    }

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model_path, metrics_path


def main():
    df = load_dataframe()
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = ["logreg", "rf", "xgb", "lgbm"]
    settings = [
        (False, False),  # no PCA, no tuning
        (False, True),   # no PCA, tuning
        (True, False),   # PCA, no tuning
        (True, True),    # PCA, tuning
    ]

    summary = []

    for model_name in models:
        for use_pca, use_tuning in settings:
            run_id = f"{model_name}__pca_{int(use_pca)}__tune_{int(use_tuning)}"
            print(f"\n=== Running: {run_id} ===")

            pipe, res, best_params = run_single(
                model_name, use_pca, use_tuning, X_train, y_train, X_test, y_test
            )

            meta = {
                "model_name": model_name,
                "use_pca": use_pca,
                "use_tuning": use_tuning,
                "best_params": best_params,
            }

            save_artifacts(run_id, pipe, res, meta)

            summary.append({
                "run_id": run_id,
                "model": model_name,
                "use_pca": use_pca,
                "use_tuning": use_tuning,
                "f1": res.f1,
            })

            print(f"✅ F1: {res.f1:.4f}")

    # Save overall summary
    summary_path = METRICS_DIR / "summary_16_experiments.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n🎉 All experiments done!")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

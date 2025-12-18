import json
from pathlib import Path
import mlflow

DAGSHUB_USER = "santhosh-madha"
DAGSHUB_REPO = "housing_app_fall25-bank-marketing"

def load_runs(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # Accept either {"runs": [...]} or just [...]
    if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
        return data["runs"]
    if isinstance(data, list):
        return data

    raise ValueError("Unknown summary JSON format. Expected a list or a dict with key 'runs'.")

def main():
    summary_path = Path("metrics") / "summary_16_experiments.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path.resolve()}")

    tracking_uri = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("bank_marketing_16_experiments")

    runs = load_runs(summary_path)
    print(f"Logging {len(runs)} runs to {tracking_uri}")

    for r in runs:
        # Try multiple key names (because your scripts might save different ones)
        run_name = r.get("run_id") or r.get("name") or r.get("id") or "run"
        model_type = r.get("model_type") or r.get("model") or r.get("algo") or "unknown"
        use_pca = int(bool(r.get("use_pca", r.get("pca", False))))
        tuned = int(bool(r.get("tuned", r.get("optuna", False))))
        f1 = float(r.get("f1", r.get("f1_score", 0.0)))

        params = r.get("params", {}) or {}

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("use_pca", use_pca)
            mlflow.set_tag("tuned", tuned)

            mlflow.log_param("model_type", model_type)
            mlflow.log_param("use_pca", use_pca)
            mlflow.log_param("tuned", tuned)

            for k, v in params.items():
                mlflow.log_param(str(k), str(v))

            mlflow.log_metric("f1", f1)

    print("✅ Done. Open the Dagshub MLflow UI to confirm your 16 runs.")

if __name__ == "__main__":
    main()

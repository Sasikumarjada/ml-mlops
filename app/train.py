"""Train an Iris classifier and save the model + metrics."""
import json
import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(output_dir: str = "model") -> dict:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "classes": list(data.target_names)}, out / "model.joblib")
    metrics = {"accuracy": round(float(acc), 4), "n_train": len(X_train), "n_test": len(X_test)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = train(sys.argv[1] if len(sys.argv) > 1 else "model")
    print(json.dumps(m))
    if m["accuracy"] < 0.9:
        sys.exit("Accuracy too low - failing the build")

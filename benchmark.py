import json
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

start_all = time.time()
base = Path.home() / "ml-benchmark"
df = pd.read_csv(base / "creditcard.csv")
load_done = time.time()

X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42,
    n_jobs=-1,
)

train_start = time.time()
model.fit(X_train, y_train)
train_end = time.time()

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

lat_start = time.time()
_ = model.predict_proba(X_test.iloc[[0]])
lat_end = time.time()

thr_start = time.time()
_ = model.predict_proba(X_test.iloc[:1000])
thr_end = time.time()

result = {
    "load_data_seconds": round(load_done - start_all, 4),
    "training_seconds": round(train_end - train_start, 4),
    "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    "auc_roc": float(round(roc_auc_score(y_test, proba), 6)),
    "accuracy": float(round(accuracy_score(y_test, pred), 6)),
    "f1_score": float(round(f1_score(y_test, pred, zero_division=0), 6)),
    "precision": float(round(precision_score(y_test, pred, zero_division=0), 6)),
    "recall": float(round(recall_score(y_test, pred, zero_division=0), 6)),
    "inference_latency_1row_ms": float(round((lat_end - lat_start) * 1000, 4)),
    "inference_throughput_1000rows_rows_per_sec": float(
        round(1000 / max((thr_end - thr_start), 1e-9), 2)
    ),
}

with open(base / "benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))

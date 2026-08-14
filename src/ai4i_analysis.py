"""
ai4i_analysis.py
-----------------
Predictive maintenance analysis on the AI4I 2020 dataset.

Trains a Decision Tree Classifier and a Support Vector Machine (RBF kernel)
to predict the binary `Machine failure` label from sensor telemetry
(air/process temperature, rotational speed, torque, tool wear) and product
quality type. Reports precision, recall, F1, and ROC-AUC rather than
accuracy alone, since failures are rare (~3.4% of records) and accuracy
alone would reward a trivial "always predict no failure" model.

Usage:
    python src/ai4i_analysis.py
"""

import json
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "ai4i_results.json")

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
TARGET = "Machine failure"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(name: str, pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(metrics["confusion_matrix"])

    return metrics


def main():
    df = load_data()
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    print(f"Total rows: {len(df)}")
    print(f"Failure rate: {100 * y.mean():.3f}%")
    print(df["Type"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    results = {}

    dt_pipe = build_pipeline(
        DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
    )
    dt_pipe.fit(X_train, y_train)
    results["Decision Tree"] = evaluate("Decision Tree", dt_pipe, X_test, y_test)

    feat_names = dt_pipe.named_steps["pre"].get_feature_names_out()
    importances = dt_pipe.named_steps["clf"].feature_importances_
    top_features = sorted(zip(feat_names, importances), key=lambda t: -t[1])[:5]
    print("\nTop feature importances (Decision Tree):")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")

    svm_pipe = build_pipeline(
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    )
    svm_pipe.fit(X_train, y_train)
    results["SVM (RBF)"] = evaluate("SVM (RBF)", svm_pipe, X_test, y_test)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

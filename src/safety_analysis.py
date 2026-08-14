"""
safety_analysis.py
-------------------
Workplace safety-incident severity analysis on the Industrial Safety and
Health Analytics Database (IHM Stefanini), a real dataset of 425 recorded
accidents across 12 industrial plants in 3 countries, predominantly in the
Mining sector.

Trains a Decision Tree Classifier and a Support Vector Machine (RBF kernel)
to predict whether a recorded accident was "severe" (Accident Level III,
IV, or V) versus minor (I or II), using categorical incident attributes
(industry sector, country, gender, employee/third-party status, critical
risk category). Reports precision, recall, F1, and ROC-AUC rather than
accuracy alone, since severe accidents are a minority class (~16%).

Usage:
    python src/safety_analysis.py
"""

import json
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "industrial_safety.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "safety_results.json")

CATEGORICAL_FEATURES = ["Industry Sector", "Countries", "Genre", "Employee or Third Party", "Critical Risk"]
SEVERE_LEVELS = ["III", "IV", "V"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["severe"] = df["Accident Level"].isin(SEVERE_LEVELS).astype(int)
    return df


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(name: str, pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
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
    X = df[CATEGORICAL_FEATURES]
    y = df["severe"]

    print(f"Total rows: {len(df)}")
    print(f"Severe accident rate: {100 * y.mean():.2f}%")
    print(df["Industry Sector"].value_counts())

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
    top_features = sorted(zip(feat_names, importances), key=lambda t: -t[1])[:6]
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

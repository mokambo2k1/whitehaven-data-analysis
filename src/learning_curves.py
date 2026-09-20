import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

def plot_learning_curve(name, X, y, pre, fname):
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    train_sizes = np.linspace(0.1, 1.0, 6)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for ax, (model_name, model) in zip(axes, models.items()):
        pipe = Pipeline([("pre", pre), ("clf", model)])
        sizes, train_scores, test_scores = learning_curve(
            pipe, X, y, cv=skf, scoring="f1", train_sizes=train_sizes, random_state=RANDOM_STATE
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

        ax.plot(sizes, train_mean, "o-", label="Training score")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        ax.plot(sizes, test_mean, "o-", label="Cross-validation score")
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
        ax.set_title(f"{name}: {model_name}")
        ax.set_xlabel("Training set size (rows)")
        ax.set_ylabel("F1 score")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")

# --- AI4I ---
df = pd.read_csv("data/ai4i2020.csv")
df.columns = [c.strip() for c in df.columns]
numeric = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
X = df[["Type"] + numeric]
y = df["Machine failure"]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]), ("num", StandardScaler(), numeric)])
plot_learning_curve("AI4I", X, y, pre, "learning_curve_ai4i.png")

# --- Safety ---
df2 = pd.read_csv("data/industrial_safety.csv")
df2["severe"] = df2["Accident Level"].isin(["III", "IV", "V"]).astype(int)
cats = ["Industry Sector", "Countries", "Genre", "Employee or Third Party", "Critical Risk"]
X2 = df2[cats]
y2 = df2["severe"]
pre2 = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
plot_learning_curve("Safety", X2, y2, pre2, "learning_curve_safety.png")

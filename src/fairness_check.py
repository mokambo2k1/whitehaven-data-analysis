import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

RANDOM_STATE = 42

def audit(name, X, y, pre, sensitive):
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    }
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sensitive, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    for model_name, model in models.items():
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        mf = MetricFrame(
            metrics={"selection_rate": selection_rate, "TPR": true_positive_rate, "FPR": false_positive_rate},
            y_true=y_te, y_pred=y_pred, sensitive_features=s_te,
        )
        print(f"\n[{name} / {model_name}] by group:")
        print(mf.by_group)
        print(f"Differences: {mf.difference().to_dict()}")

# --- AI4I: no demographic column, so use equipment quality Type as the group ---
df = pd.read_csv("data/ai4i2020.csv")
df.columns = [c.strip() for c in df.columns]
numeric = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
X = df[["Type"] + numeric]
y = df["Machine failure"]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]), ("num", StandardScaler(), numeric)])
audit("AI4I (sensitive=Type)", X, y, pre, df["Type"])

# --- Safety: Genre (gender) is the natural sensitive attribute ---
df2 = pd.read_csv("data/industrial_safety.csv")
df2["severe"] = df2["Accident Level"].isin(["III", "IV", "V"]).astype(int)
cats = ["Industry Sector", "Countries", "Genre", "Employee or Third Party", "Critical Risk"]
X2 =

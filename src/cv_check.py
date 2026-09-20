import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

def check(name, X, y, pre):
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for model_name, model in models.items():
        pipe = Pipeline([("pre", pre), ("clf", model)])
        cv = cross_validate(pipe, X, y, cv=skf, scoring="roc_auc")

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
        pipe.fit(X_tr, y_tr)
        single_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])

        print(f"{name} / {model_name}: CV ROC-AUC = {cv['test_roc_auc'].mean():.3f} "
              f"(+/- {cv['test_roc_auc'].std():.3f}) | single-split ROC-AUC = {single_auc:.3f}")

# --- AI4I ---
df = pd.read_csv("data/ai4i2020.csv")
df.columns = [c.strip() for c in df.columns]
numeric = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
X = df[["Type"] + numeric]
y = df["Machine failure"]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]), ("num", StandardScaler(), numeric)])
check("AI4I", X, y, pre)

# --- Safety ---
df2 = pd.read_csv("data/industrial_safety.csv")
df2["severe"] = df2["Accident Level"].isin(["III", "IV", "V"]).astype(int)
cats = ["Industry Sector", "Countries", "Genre", "Employee or Third Party", "Critical Risk"]
X2 = df2[cats]
y2 = df2["severe"]
pre2 = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
check("Safety", X2, y2, pre2)

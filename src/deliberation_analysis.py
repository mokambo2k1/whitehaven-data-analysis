from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

def cv_vs_single_split(X, y, pre, models, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {"f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc"}
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("clf", model)])
        cv_scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring)

        # original Task 1 single split, same seed
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=random_state
        )
        pipe.fit(X_tr, y_tr)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        single_auc = roc_auc_score(y_te, y_proba)

        results[name] = {
            "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": cv_scores["test_roc_auc"].std(),
            "single_split_roc_auc": single_auc,
        }
    return results


import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import  SVC
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_validate
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,

)

# Config
TABLE1_PATH = Path("../data/table_1.csv")
TABLE2_PATH = Path("../data/table_2.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS =  5# CV folds
#N_REPEATS = 4 # if no groups in repeatedStratifiedKFold

# Load & Prepare
def load_and_prepare(t1_path: Path, t2_path: Path):

    """Load CSVs, deduplicate IDs, merge, and build binary target."""
    if not t1_path.exists() or not t2_path.exists():
        sys.exit("Error: Missing CSV file(s). Place table_1.csv and table_2.csv in the same folder.")

    t1 = pd.read_csv(t1_path, sep=";").drop_duplicates(subset=["ID"], keep="first")
    t2 = pd.read_csv(t2_path, sep=";").drop_duplicates(subset=["ID"], keep="first")

    df = pd.merge(t1, t2, on="ID", how="inner")
    df["Type_binary"] = df["Type"].map({"y": 1, "n": 0}).astype(int)

    X = df.drop(columns=["ID", "Type", "Type_binary"])
    y = df["Type_binary"].to_numpy()
    groups = df["ID"].to_numpy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return df, X, y, groups, numeric_cols, categorical_cols


# Preprocessing

def build_preprocessor(num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocess

# ---------------------------
# CV Strategy (stratified, with groups if available)
# ---------------------------
from sklearn.model_selection import StratifiedKFold

def build_cv(y, n_splits=8, n_repeats=2, random_state=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# ---------------------------
# Train / Evaluate (hold-out)
# ---------------------------
def grouped_train_test_split(X, y, groups, test_size, random_state):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]

def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.8).astype(int)

    report_txt = classification_report(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    return report_txt, cm, roc, pr

# ---------------------------
# Cross-validated scoring
# ---------------------------
def run_cv(name, model, X, y, groups, cv):
    """
    Runs cross-validated evaluation with metrics that are robust to imbalance.
    """
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_acc": "balanced_accuracy",
        "f1": "f1",
    }
    # cross_validate will respect groups if the splitter supports groups and you pass groups=...
    kwargs = {"cv": cv, "scoring": scoring, "n_jobs": -1, "return_train_score": False}
    if isinstance(cv, StratifiedGroupKFold):
        res = cross_validate(model, X, y, groups=groups, **kwargs)
    else:
        res = cross_validate(model, X, y, **kwargs)

    def m(key):
        return np.mean(res[f"test_{key}"]), np.std(res[f"test_{key}"])

    roc_m, roc_s = m("roc_auc")
    pr_m, pr_s = m("pr_auc")
    ba_m, ba_s = m("balanced_acc")
    f1_m, f1_s = m("f1")

    print(f"\n=== {name}: {cv.__class__.__name__} ({getattr(cv, 'n_splits', 'n/a')} folds) ===")
    print(f"ROC-AUC: {roc_m:.4f} ± {roc_s:.4f}")
    print(f"PR-AUC:  {pr_m:.4f} ± {pr_s:.4f}")
    print(f"BalAcc:  {ba_m:.4f} ± {ba_s:.4f}")
    print(f"F1:      {f1_m:.4f} ± {f1_s:.4f}")

# ---------------------------
# Main
# ---------------------------
def main():
    df, X, y, groups, num_cols, cat_cols = load_and_prepare(TABLE1_PATH, TABLE2_PATH)
    preprocess = build_preprocessor(num_cols, cat_cols)
    cv = build_cv(y, n_splits=N_SPLITS, random_state=RANDOM_STATE)

    # Pipelines


    logreg = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    rf = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    svm = Pipeline([
        ("preprocess", preprocess),
        ("clf", SVC(
            kernel="rbf",
            probability=True,  # needed for ROC/PR
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    #Cross-validated evaluation
    run_cv("Logistic Regression", logreg, X, y, groups, cv)
    run_cv("Random Forest",       rf,     X, y, groups, cv)
    run_cv("SVM (RBF)",           svm,    X, y, groups, cv)

    #final group-aware hold-out for a realistic “last check”
    X_train, X_test, y_train, y_test = grouped_train_test_split(X, y, groups, TEST_SIZE, RANDOM_STATE)

    logreg.fit(X_train, y_train)
    report, cm, roc, pr = evaluate(logreg, X_test, y_test)
    print("\n=== Hold-out: Logistic Regression ===")
    print(report)
    print("ROC-AUC:", round(roc, 4), "PR-AUC:", round(pr, 4))
    print("Confusion matrix:\n", cm)

    rf.fit(X_train, y_train)
    report, cm, roc, pr = evaluate(rf, X_test, y_test)
    print("\n=== Hold-out: Random Forest ===")
    print(report)
    print("ROC-AUC:", round(roc, 4), "PR-AUC:", round(pr, 4))
    print("Confusion matrix:\n", cm)

    svm.fit(X_train, y_train)
    report, cm, roc, pr = evaluate(svm, X_test, y_test)
    print("\n=== Hold-out: SVM (RBF) ===")
    print(report)
    print("ROC-AUC:", round(roc, 4), "PR-AUC:", round(pr, 4))
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
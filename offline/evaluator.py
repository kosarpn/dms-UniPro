import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ml.classifiers.rule_based_classifier import RuleBasedClassifier
from ml.classifiers.svm_classifier import SVMClassifier
from ml.classifiers.rf_classifier import RFClassifier


# ==========================================
# Load Dataset
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

df = pd.read_csv(DATASET_PATH)


FEATURES = [
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "is_head_down",
    "is_head_left",
    "is_head_right",
]

X = df[FEATURES]
y = df["label"]


# ==========================================
# Same Train/Test Split used in training
# ==========================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ==========================================
# Rule-Based
# ==========================================

rule = RuleBasedClassifier()

rule_pred = rule.predict_batch(X_test)

print("=" * 40)
print("Rule Based")
print("=" * 40)
print("Accuracy :", accuracy_score(y_test, rule_pred))
print("Precision:", precision_score(y_test, rule_pred))
print("Recall   :", recall_score(y_test, rule_pred))
print("F1       :", f1_score(y_test, rule_pred))


# ==========================================
# SVM
# ==========================================

svm = SVMClassifier()

svm_pred = svm.predict_batch(X_test)

print("\n" + "=" * 40)
print("SVM")
print("=" * 40)
print("Accuracy :", accuracy_score(y_test, svm_pred))
print("Precision:", precision_score(y_test, svm_pred))
print("Recall   :", recall_score(y_test, svm_pred))
print("F1       :", f1_score(y_test, svm_pred))


# ==========================================
# Random Forest
# ==========================================

rf = RFClassifier()
rf_pred = rf.predict_batch(X_test)
print("\n" + "=" * 40)
print("Random Forest")
print("=" * 40)
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1       :", f1_score(y_test, rf_pred))
results = pd.DataFrame([
    {
        "Model": "Rule-Based",
        "Accuracy": accuracy_score(y_test, rule_pred),
        "Precision": precision_score(y_test, rule_pred),
        "Recall": recall_score(y_test, rule_pred),
        "F1": f1_score(y_test, rule_pred),
    },
    {
        "Model": "SVM",
        "Accuracy": accuracy_score(y_test, svm_pred),
        "Precision": precision_score(y_test, svm_pred),
        "Recall": recall_score(y_test, svm_pred),
        "F1": f1_score(y_test, svm_pred),
    },
    {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred),
        "Recall": recall_score(y_test, rf_pred),
        "F1": f1_score(y_test, rf_pred),
    },
])

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)

results.to_csv(
    results_dir / "evaluation_results.csv",
    index=False
)

print(results)
# core/evaluator.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import learning_curve
from sklearn.dummy import DummyClassifier, DummyRegressor
import matplotlib.pyplot as plt


# --------------------------------------------------
# Classification Evaluation
# --------------------------------------------------
def evaluate_classification(model, X_train, y_train, X_test, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train F1": f1_score(y_train, y_train_pred, average="weighted"),
        "Test F1": f1_score(y_test, y_test_pred, average="weighted"),
        "Precision": precision_score(y_test, y_test_pred, average="weighted"),
        "Recall": recall_score(y_test, y_test_pred, average="weighted"),
    }

    # ROC AUC (only for binary classification)
    if len(np.unique(y_test)) == 2:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["ROC AUC"] = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_test_pred)

    # Overfitting detection
    gap = metrics["Train Accuracy"] - metrics["Test Accuracy"]
    overfitting_msg = detect_overfitting(gap)

    return metrics, cm, overfitting_msg


# --------------------------------------------------
# Regression Evaluation
# --------------------------------------------------
def evaluate_regression(model, X_train, y_train, X_test, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

    gap = metrics["Train R2"] - metrics["Test R2"]
    overfitting_msg = detect_overfitting(gap)

    return metrics, overfitting_msg


# --------------------------------------------------
# Learning Curve
# --------------------------------------------------
def plot_learning_curve(model, X, y):

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1
    )

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
    ax.plot(train_sizes, test_scores.mean(axis=1), label="Validation Score")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Score")
    ax.legend()

    return fig


# --------------------------------------------------
# Overfitting Detector
# --------------------------------------------------
def detect_overfitting(gap):

    if gap > 0.1:
        return "⚠️ Possible Overfitting Detected (Large Train-Test Gap)"
    elif gap < -0.05:
        return "⚠️ Possible Underfitting Detected"
    return "✅ Model looks stable"


# --------------------------------------------------
# Baseline Model
# --------------------------------------------------
def get_baseline_score(task, X_train, y_train, X_test, y_test):

    if task == "Classification":
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        return accuracy_score(y_test, baseline.predict(X_test))

    else:
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        return r2_score(y_test, baseline.predict(X_test))
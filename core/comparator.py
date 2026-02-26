# core/comparator.py

import pandas as pd


def build_comparison_table(
    trained_models,
    task_type,
    X_train,
    y_train,
    X_test,
    y_test
):

    from core.evaluator import evaluate_classification, evaluate_regression

    results = []

    for name, model in trained_models.items():

        if task_type == "Classification":

            metrics, _, _ = evaluate_classification(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            row = {
                "Model": name,
                "Train Accuracy": metrics["Train Accuracy"],
                "Test Accuracy": metrics["Test Accuracy"],
                "Test F1": metrics["Test F1"],
            }

        else:

            metrics, _ = evaluate_regression(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            row = {
                "Model": name,
                "Train R2": metrics["Train R2"],
                "Test R2": metrics["Test R2"],
                "RMSE": metrics["RMSE"],
            }

        results.append(row)

    df = pd.DataFrame(results)

    # Ranking by TEST performance only
    if task_type == "Classification":
        df = df.sort_values(by="Test Accuracy", ascending=False)
    else:
        df = df.sort_values(by="Test R2", ascending=False)

    df.reset_index(drop=True, inplace=True)

    return df
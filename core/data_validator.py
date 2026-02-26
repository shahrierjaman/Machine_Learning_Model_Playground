# core/data_validator.py

import pandas as pd
import numpy as np


def validate_dataset(df: pd.DataFrame):

    report = {}

    report["num_rows"] = df.shape[0]
    report["num_columns"] = df.shape[1]

    # Missing values
    missing_per_column = df.isnull().sum()
    total_missing = missing_per_column.sum()

    report["missing_values"] = missing_per_column
    report["total_missing"] = int(total_missing)

    # Duplicate rows
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Constant columns
    report["constant_columns"] = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]

    # Data types
    report["data_types"] = df.dtypes

    # Numeric & categorical columns
    report["numeric_columns"] = df.select_dtypes(include=np.number).columns.tolist()
    report["categorical_columns"] = df.select_dtypes(exclude=np.number).columns.tolist()

    # Memory usage
    report["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Missing percentage
    report["missing_percentage"] = (
        (missing_per_column / len(df)) * 100
    ).round(2)

    return report
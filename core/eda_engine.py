# core/eda_engine.py

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor


def basic_statistics(df):

    stats = df.describe(include="all").T
    numeric_stats = df.describe().T

    stats["skewness"] = df.skew(numeric_only=True)
    stats["kurtosis"] = df.kurtosis(numeric_only=True)

    return stats


def missing_summary(df):

    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100

    return pd.DataFrame({
        "Missing Count": missing,
        "Missing %": percent.round(2)
    })


def feature_distribution_summary(df):

    numeric_df = df.select_dtypes(include=np.number)

    summary = pd.DataFrame({
        "Skewness": numeric_df.skew(),
        "Kurtosis": numeric_df.kurtosis(),
        "Zero Count": (numeric_df == 0).sum(),
        "Zero %": ((numeric_df == 0).sum() / len(df)) * 100
    })

    return summary.round(3)


def correlation_matrix(df):
    return df.corr(numeric_only=True)


def high_correlation_pairs(corr_matrix, threshold=0.9):

    pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                pairs.append({
                    "Feature 1": corr_matrix.columns[i],
                    "Feature 2": corr_matrix.columns[j],
                    "Correlation": val
                })

    return pd.DataFrame(pairs)


def feature_target_correlation(corr_matrix, target):

    if target not in corr_matrix.columns:
        return pd.DataFrame()

    target_corr = corr_matrix[target].drop(target)
    target_corr = target_corr.sort_values(ascending=False)

    return target_corr.to_frame(name="Correlation with Target")


def calculate_vif(df):

    numeric_df = df.select_dtypes(include=np.number).dropna()

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numeric_df.values, i)
        for i in range(numeric_df.shape[1])
    ]

    return vif_data


def detect_outliers_iqr(df):

    numeric_df = df.select_dtypes(include=np.number)
    report = []

    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]

        report.append({
            "Feature": col,
            "Outlier Count": len(outliers),
            "Outlier %": (len(outliers) / len(df)) * 100
        })

    return pd.DataFrame(report).round(2)


def detect_class_imbalance(df, target):
    distribution = df[target].value_counts(normalize=True) * 100
    return distribution.round(2)


def generate_recommendations(df, target, task_type):

    recommendations = []

    # Missing values
    if df.isnull().sum().sum() > 0:
        recommendations.append("Dataset contains missing values. Consider imputation.")

    # Skewness
    skewness = df.skew(numeric_only=True)
    if any(abs(skewness) > 1):
        recommendations.append("Highly skewed numeric features detected. Consider transformation.")

    # Multicollinearity
    corr = df.corr(numeric_only=True)
    if len(corr) > 1:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        if any(abs(upper.stack()) > 0.9):
            recommendations.append("High multicollinearity detected (>0.9 correlation).")

    # Class imbalance
    if task_type == "Classification":
        distribution = df[target].value_counts(normalize=True)
        if any(distribution < 0.2):
            recommendations.append("Class imbalance detected. Consider resampling techniques.")

    # High missing columns
    missing_percent = df.isnull().mean()
    if any(missing_percent > 0.4):
        recommendations.append("Some features have >40% missing values.")

    return recommendations
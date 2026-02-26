# core/preprocessing.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    PolynomialFeatures
)
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# ---------------------------------------------------
# Rare Category Grouper
# ---------------------------------------------------
def group_rare_categories(df, categorical_cols, threshold=0.01):

    df_copy = df.copy()

    for col in categorical_cols:
        freq = df_copy[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        df_copy[col] = df_copy[col].replace(rare_categories, "Rare")

    return df_copy


# ---------------------------------------------------
# Custom Frequency Encoder
# ---------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freq
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps[col])
        return X


# ---------------------------------------------------
# Main Pipeline Builder
# ---------------------------------------------------
def build_preprocessing_pipeline(
    df,
    target_column,
    missing_strategy,
    encoding_strategy,
    scaling_strategy,
    test_size,
    random_state,
    rare_threshold=0.01,
    use_polynomial=False,
    use_variance_threshold=False
):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(include='number').columns.tolist()
    categorical_features = X.select_dtypes(exclude='number').columns.tolist()

    # Rare grouping BEFORE split
    if categorical_features:
        X = group_rare_categories(X, categorical_features, rare_threshold)

    # ---------------------------------------------------
    # Numeric Pipeline
    # ---------------------------------------------------
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=missing_strategy))
    ]

    scaler = get_scaler(scaling_strategy)
    if scaler != "passthrough":
        numeric_steps.append(("scaler", scaler))

    if use_polynomial:
        numeric_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    # ---------------------------------------------------
    # Categorical Pipeline
    # ---------------------------------------------------
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ]

    encoder = get_encoder(encoding_strategy)

    if encoder != "passthrough":
        categorical_steps.append(("encoder", encoder))

    categorical_pipeline = Pipeline(steps=categorical_steps)

    # ---------------------------------------------------
    # Column Transformer
    # ---------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    # ---------------------------------------------------
    # Train-Test Split (before fitting)
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() < 10 else None
    )

    # ---------------------------------------------------
    # Optional Feature Selection
    # ---------------------------------------------------
    if use_variance_threshold:
        selector = VarianceThreshold(threshold=0.01)
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("variance_selector", selector)
        ])
    else:
        full_pipeline = preprocessor

    return full_pipeline, X_train, X_test, y_train, y_test


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def get_scaler(strategy):

    if strategy == "StandardScaler":
        return StandardScaler()
    elif strategy == "MinMaxScaler":
        return MinMaxScaler()
    elif strategy == "RobustScaler":
        return RobustScaler()
    elif strategy == "PowerTransformer":
        return PowerTransformer(method="yeo-johnson")
    else:
        return "passthrough"


def get_encoder(strategy):

    if strategy == "OneHot":
        return OneHotEncoder(handle_unknown="ignore")
    elif strategy == "Ordinal":
        return OrdinalEncoder()
    elif strategy == "Frequency":
        return FrequencyEncoder()
    elif strategy == "Target":
        return TargetEncoder()
    else:
        return "passthrough"
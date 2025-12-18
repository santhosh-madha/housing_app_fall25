from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]


DEFAULT_FEATURE_SPEC = FeatureSpec(
    numeric_cols=[
        "age", "duration", "campaign", "pdays", "previous",
        "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
    ],
    categorical_cols=[
        "job", "marital", "education", "default_flag", "housing_flag", "loan_flag",
        "contact", "day_of_week", "month", "poutcome",
    ],
)


def build_preprocessor(spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> ColumnTransformer:
    cat = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", spec.numeric_cols),
            ("cat", cat, spec.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def build_feature_pipeline(use_pca: bool, pca_n_components: Optional[int] = None) -> Pipeline:
    """
    Returns a pipeline that outputs a numeric feature matrix ready for a classifier.
    If use_pca=True, applies PCA after preprocessing.
    """
    steps = [("preprocess", build_preprocessor())]

    if use_pca:
        # If n_components is None, keep enough components to explain 95% variance
        if pca_n_components is None:
            steps.append(("pca", PCA(n_components=0.95, random_state=42)))
        else:
            steps.append(("pca", PCA(n_components=pca_n_components, random_state=42)))

    return Pipeline(steps=steps)

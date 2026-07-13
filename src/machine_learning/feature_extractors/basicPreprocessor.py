from itertools import product
from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

import numpy as np


class GeneralPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            fighter_1_prefix="Fighter_1_", 
            fighter_2_prefix="Fighter_2_",
            numeric_features=["Height", "Reach", "Age"],
            categorical_features=["Stance"],
        ):
        self.fighter_1_prefix = fighter_1_prefix
        self.fighter_2_prefix = fighter_2_prefix

        self.mean_imputer = SimpleImputer(strategy="mean")
        # TODO: SEE HOW TO FIX IT. MAYBE REPLACE Unknown with NaN and remove the missng_values param.
        self.most_frequent_imputer = SimpleImputer(missing_values="Unknown", strategy="most_frequent")

        self.numeric_features = [f"{prefix}{feature}" for prefix, feature in product([fighter_1_prefix, fighter_2_prefix], numeric_features)]
        self.categorical_features = [f"{prefix}{feature}" for prefix, feature in product([fighter_1_prefix, fighter_2_prefix], categorical_features)]

    def fit(
        self: Self, 
        X: pd.DataFrame, 
        y: pd.Series | None = None
    ):
        self.mean_imputer.fit(X[self.numeric_features])
        self.most_frequent_imputer.fit(X[self.categorical_features])

        return self

    def transform(self: Self, X: pd.DataFrame):
        X_transformed = X.copy()

        X_transformed[self.numeric_features] = self.mean_imputer.transform(
            X[self.numeric_features]
        )
        X_transformed[self.categorical_features] = self.most_frequent_imputer.transform(
            X[self.categorical_features]
        )

        return X_transformed

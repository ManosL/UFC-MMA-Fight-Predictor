from itertools import product
from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

import numpy as np

from ml_helpers.features import compute_fighter_feature_names
from feature_extractors.constants import (
    FIGHTER_1_PREFIX,
    FIGHTER_2_PREFIX,
    PER_FIGHTER_NUMERIC_COLUMNS,
    DIFFERENCE_FEATURES_SUFFIX,
)

class DifferenceDatasetTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fighter_1_prefix: list[str] = FIGHTER_1_PREFIX,
        fighter_2_prefix: list[str] = FIGHTER_2_PREFIX,
        to_difference_features: list[str] = PER_FIGHTER_NUMERIC_COLUMNS,
        difference_features_suffix: list[str] = DIFFERENCE_FEATURES_SUFFIX,
    ) -> None:
        self.fighter_1_prefix = fighter_1_prefix
        self.fighter_2_prefix = fighter_2_prefix
        self.to_difference_features = to_difference_features
        self.difference_features_suffix = difference_features_suffix

    def fit(
        self: Self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Self:
        self._is_fitted = True
        return self

    def transform(
        self: Self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        X_transformed = X.copy()

        for feature in self.to_difference_features:
            fighter_1_feature_name = f"{self.fighter_1_prefix}{feature}"
            fighter_2_feature_name = f"{self.fighter_2_prefix}{feature}"
            difference_feature_name = f"{feature}{self.difference_features_suffix}"

            self.__validate_feature(X_transformed, fighter_1_feature_name)
            self.__validate_feature(X_transformed, fighter_2_feature_name)

            X_transformed[difference_feature_name] = \
                X_transformed[fighter_1_feature_name] - X_transformed[fighter_2_feature_name]

            X_transformed.drop(
                [fighter_1_feature_name, fighter_2_feature_name],
                axis=1,
                inplace=True
            )

        return X_transformed

    def __validate_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
    ) -> None:
        if feature_name not in df.columns:
            raise ValueError(f"Column {feature_name} does not exists in the given DataFrame. Columns: {df.columns}")

        nan_values_number = df[feature_name].isna().sum()

        if nan_values_number > 0:
            raise ValueError(f"Column {feature_name} contains {nan_values_number} rows with NaN values.")

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

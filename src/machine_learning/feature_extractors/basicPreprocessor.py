from itertools import product
from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError

import numpy as np

from ml_helpers.features import compute_fighter_feature_names
from feature_extractors.constants import (
    FIGHTER_1_PREFIX,
    FIGHTER_2_PREFIX,
    COMMON_CATEGORICAL_FEATURES,
    COMMON_PERCENTAGE_FEATURES,
    PER_FIGHTER_CATEGORICAL_FEATURES,
    PER_FIGHTER_PERCENTAGE_FEATURES
)

class GeneralPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fighter_1_prefix: str = FIGHTER_1_PREFIX,
        fighter_2_prefix: str = FIGHTER_2_PREFIX,
        common_categorical_features: list[str] = COMMON_CATEGORICAL_FEATURES,
        per_fighter_categorical_features: list[str] = PER_FIGHTER_CATEGORICAL_FEATURES,
        common_numeric_features_to_impute: list[str] | None = None,
        per_fighter_numeric_features_to_impute: list[str] = ["Height", "Reach", "Age"],
        common_categorical_features_to_impute: list[str] | None = None,
        per_fighter_categorical_features_to_impute: list[str] = ["Stance"],
        common_percentage_features: list[str] = COMMON_PERCENTAGE_FEATURES,
        per_fighter_percentage_features: list[str] = PER_FIGHTER_PERCENTAGE_FEATURES,
    ) -> None:
        self.fighter_1_prefix = fighter_1_prefix
        self.fighter_2_prefix = fighter_2_prefix
        self.common_categorical_features = common_categorical_features
        self.per_fighter_categorical_features = per_fighter_categorical_features
        self.common_numeric_features_to_impute = common_numeric_features_to_impute
        self.per_fighter_numeric_features_to_impute = per_fighter_numeric_features_to_impute
        self.common_categorical_features_to_impute = common_categorical_features_to_impute
        self.per_fighter_categorical_features_to_impute = per_fighter_categorical_features_to_impute
        self.common_percentage_features = common_percentage_features
        self.per_fighter_percentage_features = per_fighter_percentage_features

        self.mean_imputer = SimpleImputer(strategy="mean")
        self.most_frequent_imputer = SimpleImputer(
            missing_values="Unknown",
            strategy="most_frequent"
        )

    def fit(
        self: Self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Self:
        self.categorical_features_ = list(self.common_categorical_features)
        self.categorical_features_ += compute_fighter_feature_names(
            fighter_1_prefix=self.fighter_1_prefix,
            fighter_2_prefix=self.fighter_2_prefix,
            feature_names=self.per_fighter_categorical_features or []
        )

        self.percentage_features_ = list(self.common_percentage_features)
        self.percentage_features_ += compute_fighter_feature_names(
            fighter_1_prefix=self.fighter_1_prefix,
            fighter_2_prefix=self.fighter_2_prefix,
            feature_names=self.per_fighter_percentage_features
        )

        self.numeric_features_to_impute_ = list(self.common_numeric_features_to_impute or [])
        self.numeric_features_to_impute_ += compute_fighter_feature_names(
            fighter_1_prefix=self.fighter_1_prefix,
            fighter_2_prefix=self.fighter_2_prefix,
            feature_names=self.per_fighter_numeric_features_to_impute
        )

        self.categorical_features_to_impute_ = list(self.common_categorical_features_to_impute or [])
        self.categorical_features_to_impute_ += compute_fighter_feature_names(
            fighter_1_prefix=self.fighter_1_prefix,
            fighter_2_prefix=self.fighter_2_prefix,
            feature_names=self.per_fighter_categorical_features_to_impute
        )

        self.mean_imputer.fit(X[self.numeric_features_to_impute_])
        self.most_frequent_imputer.fit(X[self.categorical_features_to_impute_])

        self._is_fitted = True
        return self

    def transform(
        self: Self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError

        X_transformed = X.copy()

        X_transformed[self.numeric_features_to_impute_] = self.mean_imputer.transform(
            X_transformed[self.numeric_features_to_impute_]
        )
        X_transformed[self.categorical_features_to_impute_] = self.most_frequent_imputer.transform(
            X_transformed[self.categorical_features_to_impute_]
        )

        for feature in self.categorical_features_:
            X_transformed[feature] = X_transformed[feature].apply(
                lambda x: x.lower() if isinstance(x, str) else x
            )

        X_transformed[self.percentage_features_] = X_transformed[self.percentage_features_] / 100.0
        X_transformed["Fight_ID"] = X_transformed["Fight_ID"].apply(lambda x: str(x))
        return X_transformed

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

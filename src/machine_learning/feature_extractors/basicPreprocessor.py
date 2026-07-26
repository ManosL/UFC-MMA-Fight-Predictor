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
    COMMON_CATEGORICAL_FEATURES,
    PER_FIGHTER_CATEGORICAL_FEATURES,
)

class GeneralPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __keep_valid_result_rows(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        valid_result_condition = X["Result"].str.lower().isin(['win', 'lose', 'draw'])
        X = pd.DataFrame(X[valid_result_condition])

        X.reset_index(drop=True, inplace=True)

        return X

    def __init__(
        self,
        fighter_1_prefix: str = FIGHTER_1_PREFIX,
        fighter_2_prefix: str = FIGHTER_2_PREFIX,
        common_categorical_features: list[str] = COMMON_CATEGORICAL_FEATURES,
        per_fighter_categorical_features: list[str] = PER_FIGHTER_CATEGORICAL_FEATURES,
        numeric_features_to_impute: list[str] = ["Height", "Reach", "Age"],
        categorical_features_to_impute: list[str] = ["Stance"],
    ) -> None:
        self.fighter_1_prefix = fighter_1_prefix
        self.fighter_2_prefix = fighter_2_prefix

        self.mean_imputer = SimpleImputer(strategy="mean")
        self.most_frequent_imputer = SimpleImputer(
            missing_values="Unknown",
            strategy="most_frequent"
        )

        self.categorical_features = common_categorical_features
        self.categorical_features += compute_fighter_feature_names(
            fighter_1_prefix=fighter_1_prefix,
            fighter_2_prefix=fighter_2_prefix,
            feature_names=per_fighter_categorical_features
        )

        self.numeric_features_to_impute = compute_fighter_feature_names(
            fighter_1_prefix=fighter_1_prefix,
            fighter_2_prefix=fighter_2_prefix,
            feature_names=numeric_features_to_impute
        )

        self.categorical_features_to_impute = compute_fighter_feature_names(
            fighter_1_prefix=fighter_1_prefix,
            fighter_2_prefix=fighter_2_prefix,
            feature_names=categorical_features_to_impute
        )

    def fit(
        self: Self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Self:
        self.mean_imputer.fit(X[self.numeric_features_to_impute])
        self.most_frequent_imputer.fit(X[self.categorical_features_to_impute])
        self._is_fitted = True

        return self

    def transform(
        self: Self,
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_transformed = X.copy()

        X_transformed = self.__keep_valid_result_rows(
            X_transformed
        )

        X_transformed[self.numeric_features_to_impute] = self.mean_imputer.transform(
            X[self.numeric_features_to_impute]
        )
        X_transformed[self.categorical_features_to_impute] = self.most_frequent_imputer.transform(
            X[self.categorical_features_to_impute]
        )

        for feature in self.categorical_features:
            X_transformed[feature] = X_transformed[feature].apply(
                lambda x: x.lower() if isinstance(x, str) else x
            )

        X_transformed["Fight_ID"] = X_transformed["Fight_ID"].apply(lambda x: str(x))
        return X_transformed

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

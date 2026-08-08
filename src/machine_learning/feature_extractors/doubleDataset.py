from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from feature_extractors.constants import (
    FIGHTER_1_PREFIX,
    FIGHTER_2_PREFIX,
    PER_FIGHTER_NUMERIC_COLUMNS,
    PER_FIGHTER_CATEGORICAL_FEATURES,
)

class DoubleDatasetTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fighter_1_prefix: str = FIGHTER_1_PREFIX,
        fighter_2_prefix: str = FIGHTER_2_PREFIX,
        to_swap_features: list[str] = PER_FIGHTER_NUMERIC_COLUMNS + PER_FIGHTER_CATEGORICAL_FEATURES,
    ) -> None:
        self.fighter_1_prefix = fighter_1_prefix
        self.fighter_2_prefix = fighter_2_prefix
        self.to_swap_features = to_swap_features

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        *,
        augment: bool = True,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X, augment=augment)

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
        *,
        augment: bool = False,
    ) -> pd.DataFrame:
        if not augment:
            return X.copy()

        if X[X["Fight_ID"].str.endswith("_doubled")].count().iloc[0] > 0:
            raise ValueError("Seems that the given dataset has already been doubled...")

        X_transformed = X.copy()

        X_tmp = X.copy()

        fighter_1_feature_names = [f"{self.fighter_1_prefix}{feature}" for feature in self.to_swap_features]
        fighter_2_feature_names = [f"{self.fighter_2_prefix}{feature}" for feature in self.to_swap_features]

        [self.__validate_feature(X, feature) for feature in fighter_1_feature_names]
        [self.__validate_feature(X, feature) for feature in fighter_2_feature_names]

        X_tmp[fighter_1_feature_names] = X[fighter_2_feature_names]
        X_tmp[fighter_2_feature_names] = X[fighter_1_feature_names]

        # The below is done in order to be able to recognize the
        # doubled rows and also be able to join between features and labels
        X_tmp["Fight_ID"] = X_tmp["Fight_ID"].apply(lambda x: f"{x}_doubled")

        X_tmp["Result"] = X_tmp["Result"].apply(lambda x: self.__revert_result(x))

        X_transformed = pd.concat([X_transformed, X_tmp], ignore_index=True)

        return X_transformed

    def __validate_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
    ) -> None:
        if feature_name not in df.columns:
            raise ValueError(f"Column {feature_name} does not exists in the given DataFrame. Columns: {df.columns}")

    def __revert_result(
        self,
        result: str,
    ) -> str:
        if result.lower() == "win":
            return "lose"

        if result.lower() == "lose":
            return "win"

        return result

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

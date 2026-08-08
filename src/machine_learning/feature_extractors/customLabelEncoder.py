from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_columns_values_map: dict[str, dict[str | bool, int]],
    ) -> None:
        self.categorical_columns_values_map = categorical_columns_values_map

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

        for col in self.categorical_columns_values_map:
            column_map = self.categorical_columns_values_map[col]

            map_keys = set(column_map.keys())
            column_unique_values = set(X[col].unique())

            not_covered_values = column_unique_values.difference(map_keys)

            if not_covered_values:
                raise ValueError(f"Map for column {col} does not cover the following values: {list(not_covered_values)}")

            X_transformed[col] = X_transformed[col].apply(lambda x: column_map[x])

        return X_transformed

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

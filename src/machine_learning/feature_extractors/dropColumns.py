from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        to_drop_columns: list[str],
    ) -> None:
        self.to_drop_columns = to_drop_columns

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


        X_transformed = X_transformed.drop(
            self.to_drop_columns,
            axis=1
        )

        return X_transformed

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

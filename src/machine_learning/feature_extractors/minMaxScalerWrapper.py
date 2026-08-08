from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError


class MinMaxScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        to_scale_features: list[str],
        feature_range: tuple[int, int] = (0, 1),
    ) -> None:
        self.to_scale_features = to_scale_features
        self.feature_range = feature_range

        self.scaler = MinMaxScaler(self.feature_range)

    def fit(
        self: Self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Self:
        self.scaler.fit(X[self.to_scale_features])
        self._is_fitted = True

        return self

    def transform(
        self: Self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError

        X_transformed = X.copy()

        X_transformed[self.to_scale_features] = self.scaler.transform(
            X_transformed[self.to_scale_features]
        )

        return X_transformed

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

import os
from itertools import product
import pandas as pd

from common.minio_utils import MinioClient


TRAIN_FILE_PREFIX = "train"
TEST_FILE_PREFIX = "test"

FEATURES_FILE_SUFFIX = "features"
LABELS_FILE_SUFFIX = "labels"


def write_dataset_instance_to_minio(
    minio_client: MinioClient,
    bucket_name: str,
    path_to_write: str,
    *,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    y_test: pd.DataFrame | None = None,
) -> None:
    to_save_dfs = [X_train, y_train, X_test, y_test]
    output_file_names = [
        f"{usage}_{content}.csv" 
        for usage, content in product(
            [TRAIN_FILE_PREFIX, TEST_FILE_PREFIX],
            [FEATURES_FILE_SUFFIX, LABELS_FILE_SUFFIX]
        )
    ]

    if (X_test is None) ^ (y_test is None):
        raise ValueError("Cannot provide only the features without the labels or vice versa.")

    if (X_train.shape[0] != y_train.shape[0]) or \
        (X_test is not None and (X_test.shape[0] != y_test.shape[0])):
        raise ValueError("Features and Labels cannot have different number of rows")
    
    if X_test is None:
        to_save_dfs = to_save_dfs[:2]
        output_file_names = output_file_names[:2]

    for df, file_name in zip(to_save_dfs, output_file_names, strict=True):
        minio_client.write_pandas_df_as_csv(
            bucket_name,
            os.path.join(path_to_write, file_name),
            df,
            sep='|',
            na_rep='NaN',
            index=False
        )

    return
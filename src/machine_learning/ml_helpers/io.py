import os
from itertools import product
import pandas as pd

from common.minio_utils import MinioClient


ORIGINAL_DATA_DIR = "original"

TRAIN_FILES_DIR = "train"
TEST_FILES_DIR = "test"

FEATURES_FILE_NAME = "features"
LABELS_FILE_NAME = "labels"


def __extract_basename_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def read_dataset_instance_from_minio(
    minio_client: MinioClient,
    bucket_name: str,
    root_path: str,
) -> dict[str, pd.DataFrame | None]:
    return_value = {
        "X_train": None,
        "y_train": None,
        "X_test": None,
        "y_test": None
    }

    root_folder_dirs = minio_client.list_bucket_directory(
        bucket_name,
        root_path,
        recursive=False
    )
    root_folder_dirs = [__extract_basename_from_path(obj._object_name) for obj in root_folder_dirs]

    if TRAIN_FILES_DIR not in root_folder_dirs:
        raise FileNotFoundError(f"{TRAIN_FILES_DIR} directory does not exists in {root_path}.")

    for key, file_name in zip(["X_train", "y_train"], [FEATURES_FILE_NAME, LABELS_FILE_NAME]):
        path_to_read = os.path.join(root_path, TRAIN_FILES_DIR, f"{file_name}.csv")

        return_value[key] = minio_client.read_csv_to_pandas(
            bucket_name,
            path_to_read,
            sep="|",
            header=0
        )

        print("Gonna read", path_to_read)

    test_exists = TEST_FILES_DIR in root_folder_dirs

    if test_exists:
        for key, file_name in zip(["X_test", "y_test"], [FEATURES_FILE_NAME, LABELS_FILE_NAME]):
            path_to_read = os.path.join(root_path, TEST_FILES_DIR, f"{file_name}.csv")

            return_value[key] = minio_client.read_csv_to_pandas(
                bucket_name,
                path_to_read,
                sep="|",
                header=0
            )

            print("Gonna read", path_to_read)
    else:
        print(f"{TEST_FILES_DIR} does not exists in {root_path}. Check if it's valid.")

    return return_value


def write_dataset_instance_to_minio(
    minio_client: MinioClient,
    bucket_name: str,
    root_path: str,
    *,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    y_test: pd.DataFrame | None = None,
) -> None:
    to_save_dfs = [X_train, y_train, X_test, y_test]
    to_save_dfs_metadata = [
        {
            "usage": usage,
            "content": content,
        } for usage, content in product(
            [TRAIN_FILES_DIR, TEST_FILES_DIR],
            [FEATURES_FILE_NAME, LABELS_FILE_NAME]
        )
    ]

    if (X_test is None) ^ (y_test is None):
        raise ValueError("Cannot provide only the features without the labels or vice versa.")

    if (X_train.shape[0] != y_train.shape[0]) or \
        (X_test is not None and (X_test.shape[0] != y_test.shape[0])):
        raise ValueError("Features and Labels cannot have different number of rows")
    
    if X_test is None:
        to_save_dfs = to_save_dfs[:2]
        to_save_dfs_metadata = to_save_dfs_metadata[:2]

    for df, df_metadata in zip(to_save_dfs, to_save_dfs_metadata, strict=True):
        minio_client.write_pandas_df_as_csv(
            bucket_name,
            os.path.join(
                root_path, 
                df_metadata["usage"], 
                f"{df_metadata['content']}.csv"
            ),
            df,
            metadata=df_metadata,
            sep='|',
            na_rep='NaN',
            index=False
        )

    return

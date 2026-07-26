import argparse
import os
import secrets
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

from common.minio_utils import MinioClient

from feature_extractors.basicPreprocessor import GeneralPreprocessingTransformer
from feature_extractors.doubleDataset import DoubleDatasetTransformer
from feature_extractors.differenceDataset import DifferenceDatasetTransformer

from ml_helpers.io import (
    ORIGINAL_DATA_DIR,
    read_dataset_instance_from_minio,
    write_dataset_instance_to_minio,
)

from feature_extractors.constants import (
    FIGHT_ID_COLUMN,
    LABEL_COLUMNS,
)


def __extract_basename_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def join_dfs(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    df = features_df.merge(
        labels_df,
        how="inner",
        on=FIGHT_ID_COLUMN,
    )

    return df


def split_dfs(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels    = df[[FIGHT_ID_COLUMN] + LABEL_COLUMNS]
    features  = df.drop(LABEL_COLUMNS, axis=1)

    return features, labels


def main(
    ml_pipeline_run_id: str,
) -> int:
    minio_client = MinioClient(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    bucket_name = os.environ.get("MINIO_ML_TRAINING_DATA_BUCKET_NAME")

    # TODO: LATER MOVE THOSE TO YAML
    # TODO: THINK OF DOING IT IN AIRFLOW IN ORDER TO HAVE THEM DONE IN PARALLEL
    # OR DO IT INSIDE THE SCRIPT IN PARALLEL
    feature_extractors = [
        # TODO: VERIFY IT WORKS
        (
            "GeneralProcessing",
            GeneralPreprocessingTransformer()
        ),
        (
            "DoubleDataset",
            Pipeline(
                [
                    ("general_preprocessing", GeneralPreprocessingTransformer()),
                    ("double_dataset", DoubleDatasetTransformer())
                ]
            )
        ),
        (
            "DifferenceDataset",
            Pipeline(
                [
                    ("general_preprocessing", GeneralPreprocessingTransformer()),
                    ("difference_dataset", DifferenceDatasetTransformer())
                ]
            )
        ),
        (
            "DoubleDifferenceDataset",
            Pipeline(
                [
                    ("general_preprocessing", GeneralPreprocessingTransformer()),
                    ("double_dataset", DoubleDatasetTransformer()),
                    ("difference_dataset", DifferenceDatasetTransformer())
                ]
            )
        ),
    ]

    data_folders_paths = [os.path.join(ml_pipeline_run_id, ORIGINAL_DATA_DIR, "all/")]
    data_folders_paths += [
        obj._object_name
        for obj in minio_client.list_bucket_directory(
            bucket_name,
            os.path.join(ml_pipeline_run_id, ORIGINAL_DATA_DIR, "splits/"),
            recursive=False
        )
    ]

    for path in data_folders_paths:
        dataset_instance = read_dataset_instance_from_minio(
            minio_client,
            bucket_name,
            path
        )

        full_train_df = join_dfs(
            dataset_instance["X_train"],
            dataset_instance["y_train"]
        )

        full_test_df = join_dfs(
            dataset_instance["X_test"],
            dataset_instance["y_test"]
        ) if dataset_instance["X_test"] is not None else None

        for extractor_name, extractor_obj in feature_extractors:
            extractor_root_path = os.path.join(ml_pipeline_run_id, extractor_name)
            extractor_dataset_path = path.replace(ORIGINAL_DATA_DIR, extractor_name)
            print("Will write to", extractor_dataset_path)

            extractor_obj.fit(full_train_df, None)

            if __extract_basename_from_path(extractor_dataset_path) == "all":
                print("Writing pickle object...")

                extractor_pickle_file_path = os.path.join("/tmp", f"{secrets.token_hex(10)}.pkl")
                with open(extractor_pickle_file_path, "wb") as file:
                    pickle.dump(extractor_obj, file)

                minio_client.write_file(
                    bucket_name,
                    extractor_pickle_file_path,
                    os.path.join(extractor_root_path, "extractor.pkl")
                )

                os.remove(extractor_pickle_file_path)

            extractor_X_train, extractor_y_train = split_dfs(
                extractor_obj.transform(
                    full_train_df
                )
            )

            if dataset_instance["X_test"] is not None:
                extractor_X_test, extractor_y_test = split_dfs(
                    extractor_obj.transform(
                        full_test_df
                    )
                )
            else:
                extractor_X_test = None
                extractor_y_test = None

            write_dataset_instance_to_minio(
                minio_client,
                bucket_name,
                extractor_dataset_path,
                X_train=extractor_X_train,
                y_train=extractor_y_train,
                X_test=extractor_X_test,
                y_test=extractor_y_test
            )

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ml_pipeline_run_id", "-v", help="Machine Learning Pipeline Run ID")
    parser.parse_args()
    main(parser.ml_pipeline_run_id)

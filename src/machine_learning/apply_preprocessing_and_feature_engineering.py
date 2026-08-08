import argparse
from copy import deepcopy
import os
import secrets
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import set_config

set_config(enable_metadata_routing=True)

from common.minio_utils import MinioClient

from feature_extractors.basicPreprocessor import GeneralPreprocessingTransformer
from feature_extractors.doubleDataset import DoubleDatasetTransformer
from feature_extractors.differenceDataset import DifferenceDatasetTransformer
from feature_extractors.dropColumns import DropColumnsTransformer
from feature_extractors.customLabelEncoder import CustomLabelEncoder
from feature_extractors.minMaxScalerWrapper import MinMaxScalerWrapper

from ml_helpers.io import (
    ORIGINAL_DATA_DIR,
    read_dataset_instance_from_minio,
    write_dataset_instance_to_minio,
)

from ml_helpers.features import (
    compute_fighter_feature_names,
)

from feature_extractors.constants import (
    FIGHT_ID_COLUMN,
    FIGHT_DATE_COLUMN,
    LABEL_COLUMNS,
    FIGHTER_1_PREFIX,
    FIGHTER_2_PREFIX,
    DIFFERENCE_FEATURES_SUFFIX,
    PER_FIGHTER_ID_COLUMNS,
    COMMON_CATEGORICAL_FEATURES,
    PER_FIGHTER_NUMERIC_COLUMNS,
    PER_FIGHTER_CATEGORICAL_FEATURES,
    PER_FIGHTER_PERCENTAGE_FEATURES,
    GENDER_MAP,
    TITLE_FIGHT_MAP,
    WEIGHT_CLASS_MAP,
    FIGHT_TIME_FORMAT_MAP,
    STANCE_MAP,
)


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

    # TODO: THINK OF DOING IT IN AIRFLOW IN ORDER TO HAVE THEM DONE IN PARALLEL
    # OR DO IT INSIDE THE SCRIPT IN PARALLEL
    feature_extractors = [
        # TODO: VERIFY IT WORKS
        (
            "GeneralProcessing",
            build_pipeline(
                to_double_dataset=False,
                to_difference_dataset=False
            ),
        ),
        (
            "DoubleDataset",
            build_pipeline(
                to_double_dataset=True,
                to_difference_dataset=False
            ),
        ),
        (
            "DifferenceDataset",
            build_pipeline(
                to_double_dataset=False,
                to_difference_dataset=True
            ),
        ),
        (
            "DoubleDifferenceDataset",
            build_pipeline(
                to_double_dataset=True,
                to_difference_dataset=True
            ),
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

            if "double_dataset" in extractor_obj.named_steps:
                extractor_X_train, extractor_y_train = split_dfs(
                    extractor_obj.transform(
                        full_train_df,
                        augment=True
                    )
                )
            else:
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


def build_pipeline(
    *,
    to_double_dataset: bool,
    to_difference_dataset: bool,
) -> Pipeline:
    pipeline_steps = [
        ("general_preprocessing", GeneralPreprocessingTransformer())
    ]

    if to_double_dataset:
        pipeline_steps.append(
            (
                "double_dataset",
                DoubleDatasetTransformer().set_transform_request(augment=True)
            )
        )

    if to_difference_dataset:
        pipeline_steps.append(
            ("difference_dataset", DifferenceDatasetTransformer())
        )

    pipeline_steps.extend(
        [
            (
                "drop_columns",
                DropColumnsTransformer(
                    to_drop_columns=__determine_columns_to_drop()
                )
            ),
            (
                "encode",
                CustomLabelEncoder(
                    categorical_columns_values_map=__determine_label_encoder_columns_map()
                )
            ),
            (
                "scaling",
                MinMaxScalerWrapper(
                    to_scale_features=__determine_features_to_scale(to_difference_dataset)
                )
            ),
        ]
    )

    return Pipeline(pipeline_steps)


def __determine_columns_to_drop() -> list[str]:
    return [
        *compute_fighter_feature_names(
            fighter_1_prefix=FIGHTER_1_PREFIX,
            fighter_2_prefix=FIGHTER_2_PREFIX,
            feature_names=PER_FIGHTER_ID_COLUMNS
        ),
        FIGHT_DATE_COLUMN
    ]


def __determine_label_encoder_columns_map() -> dict[str, dict[str | bool, int]]:
    return {
        col_name: col_map
        for col_name, col_map in zip(
            COMMON_CATEGORICAL_FEATURES,
            [GENDER_MAP, WEIGHT_CLASS_MAP, TITLE_FIGHT_MAP, FIGHT_TIME_FORMAT_MAP]
        )
    } | {
        col_name: STANCE_MAP
        for col_name in compute_fighter_feature_names(
            fighter_1_prefix=FIGHTER_1_PREFIX,
            fighter_2_prefix=FIGHTER_2_PREFIX,
            feature_names=PER_FIGHTER_CATEGORICAL_FEATURES
        )
    }


def __determine_features_to_scale(
    to_difference_dataset: bool
) -> list[str]:
    if not to_difference_dataset:
        return list(
            compute_fighter_feature_names(
                fighter_1_prefix=FIGHTER_1_PREFIX,
                fighter_2_prefix=FIGHTER_2_PREFIX,
                feature_names=[
                    col
                    for col in PER_FIGHTER_NUMERIC_COLUMNS
                    if col not in set(PER_FIGHTER_PERCENTAGE_FEATURES)
                ]
            )
        )

    return [
        f"{col}{DIFFERENCE_FEATURES_SUFFIX}"
        for col in PER_FIGHTER_NUMERIC_COLUMNS
        if col not in set(PER_FIGHTER_PERCENTAGE_FEATURES)
    ]


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
        validate="one_to_one",
    )

    return df


def split_dfs(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels    = df[[FIGHT_ID_COLUMN] + LABEL_COLUMNS]
    features  = df.drop(LABEL_COLUMNS, axis=1)

    return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ml_pipeline_run_id", "-v", help="Machine Learning Pipeline Run ID")
    args = parser.parse_args()
    main(args.ml_pipeline_run_id)

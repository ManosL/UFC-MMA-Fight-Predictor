import sys
import os
import argparse
import itertools
import mlflow

from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import SVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score

from common.minio_utils import MinioClient
from feature_extractors.constants import FIGHT_ID_COLUMN
from ml_helpers.io import ORIGINAL_DATA_DIR
from ml_helpers.io import (
    extract_basename_from_path,
    read_dataset_instance_from_minio,
    read_folds_from_dataset_instance,
    read_feature_extractor_from_dataset_instance
)

TRUSTED_TYPES = [
    'feature_extractors.basicPreprocessor.GeneralPreprocessingTransformer',
    'feature_extractors.customLabelEncoder.CustomLabelEncoder',
    'feature_extractors.differenceDataset.DifferenceDatasetTransformer',
    'feature_extractors.dropColumns.DropColumnsTransformer',
    'feature_extractors.minMaxScalerWrapper.MinMaxScalerWrapper',
    'numpy.dtype',
]

# TODO: DO EACH ELEMENT AS A CLASS(USE PYDANTIC)
# WHICH CAN HAVE A GENERATOR METHOD TO YIELD THE AVAILABLE
# EXPERIMENTS.
EXPERIMENTS_HYPERPARAMETERS = [
    (
        LogisticRegression,
        {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "max_iter": [5000],
        },
    ),
    (
        KNeighborsClassifier,
        {
            "n_neighbors": [3, 5, 11, 21, 51],
            "weights": ["uniform", "distance"],
            "p": [1, 2],  # Manhattan vs Euclidean
        },
    ),
    (
        SVC,
        {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto"],
        },
    ),
    (
        DecisionTreeClassifier,
        {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10],
        },
    ),
    (
        RandomForestClassifier,
        {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", "log2"],
        },
    ),
    (
        AdaBoostClassifier,
        {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
        },
    ),
]


def main(
    ml_pipeline_run_id: str,
    feature_extractor_name: str,
    experiment_name: str,
) -> int:
    minio_client = MinioClient(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    bucket_name = os.environ.get("MINIO_ML_TRAINING_DATA_BUCKET_NAME")

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    print(f"ML Pipeline Run ID: {ml_pipeline_run_id}")
    print(f"Feature Extractor Name: {feature_extractor_name}")
    print(f"Experiment Name: {experiment_name}")
    sys.stdout.flush()

    processed_data_dir_path = os.path.join(ml_pipeline_run_id, feature_extractor_name)

    folds_dir_path = os.path.join(processed_data_dir_path, "splits/")

    all_data = read_dataset_instance_from_minio(
        minio_client,
        bucket_name,
        os.path.join(processed_data_dir_path, "all/")
    )

    all_data = prepare_fold_data_for_experiment(all_data)

    folds = read_folds_from_dataset_instance(
        minio_client,
        bucket_name,
        folds_dir_path
    )

    folds = [prepare_fold_data_for_experiment(fold_data) for fold_data in folds]

    extractor_obj = read_feature_extractor_from_dataset_instance(
        minio_client,
        bucket_name,
        processed_data_dir_path
    )

    for clf_ref, clf_hyperparameters_array in EXPERIMENTS_HYPERPARAMETERS:
        hyperparameters_experiments = [
            dict(zip(clf_hyperparameters_array.keys(), values))
            for values in itertools.product(*clf_hyperparameters_array.values())
        ]

        for experiment in hyperparameters_experiments:
            run_name = __get_clf_str_repr(clf_ref, experiment)
            run_name = f"{feature_extractor_name} + {run_name}"

            with mlflow.start_run(
                run_name=run_name,
                experiment_id=experiment["experiment_id"]
            ):
                print(f"Started {run_name}")
                mlflow.log_params({
                    "feature_extractor": feature_extractor_name,
                    "classifier": clf_ref.__name__,
                })
                sys.stdout.flush()
                clf_object = clf_ref(**experiment)

                train_accuracy_scores = []
                val_accuracy_scores = []

                for i, fold in enumerate(folds):
                    with mlflow.start_run(
                        run_name=f"fold_{i+1}",
                        experiment_id=experiment["experiment_id"],
                        nested=True
                    ):
                        train_ids, test_ids, X_train, X_test, y_train, y_test = fold

                        clf_object.fit(X_train, y_train)
                        train_predictions = clf_object.predict(X_train)
                        test_predictions = clf_object.predict(X_test)

                        train_accuracy = accuracy_score(y_train, train_predictions)
                        val_accuracy = accuracy_score(y_test, test_predictions)

                        log_detailed_results(
                            train_ids,
                            y_train,
                            train_predictions,
                            "train.csv"
                        )

                        log_detailed_results(
                            test_ids,
                            y_test,
                            test_predictions,
                            "test.csv"
                        )

                        mlflow.log_metric("train_accuracy", train_accuracy)
                        mlflow.log_metric("val_accuracy", val_accuracy)

                        train_accuracy_scores.append(train_accuracy)
                        val_accuracy_scores.append(val_accuracy)

                _, _, X_train, X_test, y_train, y_test = all_data

                clf_object.fit(X_train, y_train)

                # TODO: MAYBE UNIFY THEM IN A SINGLE PIPELINE
                mlflow.sklearn.log_model(sk_model=extractor_obj, name="extractor", serialization_format="cloudpickle",)
                mlflow.sklearn.log_model(sk_model=clf_object, name="predictor")

                mlflow.log_metric("mean_train_accuracy", np.mean(train_accuracy_scores))
                mlflow.log_metric("std_train_accuracy", np.std(train_accuracy_scores))
                mlflow.log_metric("mean_val_accuracy", np.mean(val_accuracy_scores))
                mlflow.log_metric("std_val_accuracy", np.std(val_accuracy_scores))

    return 0


def __get_clf_str_repr(clf_ref, clf_kwargs):
    arg_list_repr = ', '.join([f"{k}={v}" for k, v in clf_kwargs.items()])

    return f"{clf_ref.__name__}({arg_list_repr})"


def prepare_fold_data_for_experiment(
    fold_data: dict[str, pd.DataFrame],
) -> tuple[
        pd.Series,
        pd.Series,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series
    ]:
    X_train = fold_data["X_train"]
    X_test = fold_data["X_test"]
    y_train = fold_data["y_train"]
    y_test = fold_data["y_test"]

    ids_train = X_train[FIGHT_ID_COLUMN]
    ids_test = X_test[FIGHT_ID_COLUMN] if X_test is not None else None

    X_train = X_train.drop(FIGHT_ID_COLUMN, axis=1)
    X_test = X_test.drop(FIGHT_ID_COLUMN, axis=1) if X_test is not None else None

    y_train = y_train["Result"]
    y_test = y_test["Result"] if y_test is not None else None

    return ids_train, ids_test, X_train, X_test, y_train, y_test


def log_detailed_results(
    ids: Sequence[str],
    actual_labels: Sequence[str | int | float],
    predicted_labels: Sequence[str | int | float],
    file_path_in_run: str,
) -> None:
    detailed_df = pd.DataFrame(
        zip(list(ids), list(actual_labels), list(predicted_labels), strict=True),
        columns=["Fight_ID", "Actual Result", "Predicted Result"]
    )

    detailed_df_content = detailed_df.to_csv(index=False)

    mlflow.log_text(
        detailed_df_content,
        artifact_file=file_path_in_run
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ml_pipeline_run_id", "-v", help="Machine Learning Pipeline Run ID")
    parser.add_argument("--feature_extractor_name", "-f", help="Feature Extractor Dir Name")
    parser.add_argument("--experiment_name", "-e", help="Number of folds in the TimeSeriesSplit")
    args = parser.parse_args()

    main(args.ml_pipeline_run_id, args.experiment_name)

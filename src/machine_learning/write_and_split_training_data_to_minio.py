import argparse
import os
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
from sklearn.model_selection import TimeSeriesSplit

from common.minio_utils import MinioClient
from common.psycopg_utils import get_postgres_connection

from ml_helpers.io import ORIGINAL_DATA_DIR
from ml_helpers.io import write_dataset_instance_to_minio

from feature_extractors.constants import FIGHT_DATE_COLUMN, LABEL_COLUMNS, FIGHT_ID_COLUMN


def read_fights_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    postgres_connection = get_postgres_connection()
    query = "SELECT * FROM \"ML_Fights\""

    fights_df = sqlio.read_sql_query(query, postgres_connection)

    postgres_connection.close()

    # Also writing in labels df the Fight_ID(the IDs in both datasets
    # are in the same order)
    labels    = fights_df[[FIGHT_ID_COLUMN] + LABEL_COLUMNS]
    features  = fights_df.drop(LABEL_COLUMNS, axis=1)

    return features, labels


def main(
    ml_pipeline_run_id: str,
    k_folds: int
) -> int:
    minio_client = MinioClient(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    bucket_name = os.environ.get("MINIO_ML_TRAINING_DATA_BUCKET_NAME")

    features, labels = read_fights_data()

    write_dataset_instance_to_minio(
        minio_client,
        bucket_name,
        os.path.join(ml_pipeline_run_id, ORIGINAL_DATA_DIR, "all"),
        X_train=features,
        y_train=labels
    )

    time_series_cv = TimeSeriesSplit(n_splits=k_folds)

    dates_series = np.sort(features[FIGHT_DATE_COLUMN].unique())

    for i, (train_dates_index, test_dates_index) in enumerate(time_series_cv.split(dates_series)):
        train_dates = dates_series[train_dates_index]
        test_dates = dates_series[test_dates_index]

        train_features = features[features[FIGHT_DATE_COLUMN].isin(train_dates)]
        train_labels = labels[features[FIGHT_DATE_COLUMN].isin(train_dates)]

        test_features = features[features[FIGHT_DATE_COLUMN].isin(test_dates)]
        test_labels = labels[features[FIGHT_DATE_COLUMN].isin(test_dates)]

        write_dataset_instance_to_minio(
            minio_client,
            bucket_name,
            os.path.join(ml_pipeline_run_id, ORIGINAL_DATA_DIR, "splits", f"split_{i+1}"),
            X_train=train_features,
            y_train=train_labels,
            X_test=test_features,
            y_test=test_labels
        )

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ml_pipeline_run_id", "-v", help="Machine Learning Pipeline Run ID")
    parser.add_argument("--folds", "-v", help="Number of folds in the TimeSeriesSplit")
    parser.parse_args()
    main(parser.ml_pipeline_run_id, parser.folds)

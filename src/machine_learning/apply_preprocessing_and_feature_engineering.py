import argparse
import os
import secrets
import pickle
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
from sklearn.model_selection import TimeSeriesSplit

from common.minio_utils import MinioClient

from feature_extractors.basicPreprocessor import GeneralPreprocessingTransformer
from ml_helpers.io import (
    ORIGINAL_DATA_DIR,
    read_dataset_instance_from_minio,
    write_dataset_instance_to_minio,
)


def __extract_basename_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


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

    feature_extractors = [
        # TODO: VERIFY IT WORKS
        ("GeneralProcessing", GeneralPreprocessingTransformer())
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

        for extractor_name, extractor_obj in feature_extractors:
            extractor_root_path = os.path.join(ml_pipeline_run_id, extractor_name)
            extractor_dataset_path = path.replace(ORIGINAL_DATA_DIR, extractor_name)
            print("Will write to", extractor_dataset_path)

            extractor_obj.fit(dataset_instance["X_train"])

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

            extractor_X_train = extractor_obj.transform(dataset_instance["X_train"])
            extractor_y_train = dataset_instance["y_train"].copy()

            if dataset_instance["X_test"] is not None:
                extractor_X_test = extractor_obj.transform(dataset_instance["X_test"])
                extractor_y_test = dataset_instance["y_test"].copy()
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

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ml_pipeline_run_id", "-v", help="Machine Learning Pipeline Run ID")
    parser.parse_args()
    main(parser.ml_pipeline_run_id)

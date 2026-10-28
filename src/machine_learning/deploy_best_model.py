import argparse
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.entities import Run

from retrieve_best_run_id import should_replace_best_run


REGISTERED_MODEL_NAME = "ufc_fight_predictor"
CHAMPION_MODEL_ALIAS = "champion"


def registered_model_exists(name: str, alias: str) -> bool:
    client = MlflowClient()

    try:
        client.get_model_version_by_alias(name, alias)
        return True
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False

        raise e


def get_current_champion_model_run() -> Run:
    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_MODEL_ALIAS
    )

    return mlflow.get_run(champion.run_id)


def main(
    run_id: str,
) -> int:
    run = mlflow.get_run(run_id)

    if registered_model_exists(REGISTERED_MODEL_NAME, CHAMPION_MODEL_ALIAS):
        champion_model_run = get_current_champion_model_run()
        
        if not should_replace_best_run(champion_model_run, run):
            print(f"Won't register/deploy the model with run_id {run_id}\n \
                because it's not better than the currently deployed one")

            return 0

    for model in run.outputs.model_outputs:
        print(model.model_id)

    model_id = run.outputs.model_outputs[0].model_id

    model_version = mlflow.register_model(
        model_uri=f"models:/{model_id}",
        name=REGISTERED_MODEL_NAME,
    )

    print(f"Registered version(also champion one): {model_version.version}")

    client = MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_MODEL_ALIAS,
        version=model_version.version
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", "-r", help="MLflow's Run ID that we want to deploy it's inference model")
    args = parser.parse_args()

    main(args.run_id)

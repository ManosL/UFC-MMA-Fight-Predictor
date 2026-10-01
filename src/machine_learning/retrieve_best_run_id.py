import argparse
import mlflow
from mlflow.entities import Run


def should_replace_best_run(
    current_best_run: Run,
    candidate_for_best_run: Run,
) -> bool:
    current_best_run_metrics = current_best_run.data.metrics
    candidate_for_best_run_metrics = candidate_for_best_run.data.metrics

    if 0 < candidate_for_best_run_metrics["mean_val_accuracy"] - current_best_run_metrics["mean_val_accuracy"] < 0.02:
        current_best_run_overfitting_rate = current_best_run_metrics["mean_train_accuracy"] - current_best_run_metrics["mean_val_accuracy"]
        candidate_for_best_run_overfitting_rate = candidate_for_best_run_metrics["mean_train_accuracy"] - candidate_for_best_run_metrics["mean_val_accuracy"]

        return candidate_for_best_run_overfitting_rate < current_best_run_overfitting_rate

    return candidate_for_best_run_metrics["mean_val_accuracy"] > current_best_run_metrics["mean_val_accuracy"]


def main(
    experiment_name: str,
) -> str:
    experiment_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attribute.status = 'FINISHED' AND tags.mlflow.parentRunId IS NULL",
        format='list'
    )

    best_run = experiment_runs[0]

    for run in experiment_runs[1:]:
        if should_replace_best_run(best_run, run):
            best_run = run

    return best_run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", "-e", help="MLflow Experiment Name")
    args = parser.parse_args()

    main(args.experiment_name)

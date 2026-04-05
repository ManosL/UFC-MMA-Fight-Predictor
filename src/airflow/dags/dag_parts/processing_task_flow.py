from collections.abc import Sequence
import sys
from airflow.models.baseoperator import BaseOperator
from airflow.operators.python import PythonVirtualenvOperator
from airflow.operators.empty import EmptyOperator


def get_processing_requirements() -> Sequence[str]:
    with open("/opt/airflow/processing/requirements.txt") as f:
        requirements = [l.strip() for l in f.readlines()
                        if not l.strip().startswith("#")]

    return requirements


def clean_and_process_fights_venv() -> None:
    sys.path.append("/opt/airflow/processing")
    from clean_and_process_fights import main

    main()
    return


def configure_fighters_dataset_venv() -> None:
    sys.path.append("/opt/airflow/processing")
    from Configure_Fighters_Dataset import main

    main()
    return


def load_to_postgres_venv() -> None:
    sys.path.append("/opt/airflow/processing")
    from load_to_postgres import main

    main()
    return


def get_processing_task_flow() -> BaseOperator:
    start_task = EmptyOperator(task_id="start_processing")

    processing_requirements = get_processing_requirements()

    clean_and_process_fights_task = PythonVirtualenvOperator(
        task_id="clean_and_process_fights",
        python_callable=clean_and_process_fights_venv,
        requirements=processing_requirements,
        system_site_packages=False,
    )

    configure_fighters_dataset_task = PythonVirtualenvOperator(
        task_id="configure_fighters_dataset",
        python_callable=configure_fighters_dataset_venv,
        requirements=processing_requirements,
        system_site_packages=False,
    )

    load_to_postgres_task = PythonVirtualenvOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres_venv,
        requirements=processing_requirements,
        system_site_packages=False,
    )

    end_task = EmptyOperator(task_id="end_processing")

    start_task >> [clean_and_process_fights_task, configure_fighters_dataset_task]
    [clean_and_process_fights_task, configure_fighters_dataset_task] >> load_to_postgres_task
    load_to_postgres_task >> end_task

    return start_task
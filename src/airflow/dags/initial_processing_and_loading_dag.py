import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
from airflow.operators.empty import EmptyOperator


def get_processing_requirements():
    with open("/opt/airflow/processing/requirements.txt") as f:
        requirements = [l.strip() for l in f.readlines()
                        if not l.strip().startswith("#")]

    return requirements


def clean_and_process_fights_venv():
    sys.path.append("/opt/airflow/processing")
    from clean_and_process_fights import main

    main()
    return


def configure_fighters_dataset_venv():
    sys.path.append("/opt/airflow/processing")
    from Configure_Fighters_Dataset import main

    main()
    return


# def configure_event_dataset_venv():
#     sys.path.append("/opt/airflow/processing")
#     from Configure_Event_Dataset import main

#     main()
#     return


def load_to_postgres_venv():
    sys.path.append("/opt/airflow/processing")
    from load_to_postgres import main

    main()
    return


with DAG(
    dag_id="initial_processing_and_loading",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    start_task = EmptyOperator(task_id="start")

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

    # configure_event_dataset_task = PythonVirtualenvOperator(
    #     task_id="configure_event_dataset",
    #     python_callable=configure_event_dataset_venv,
    #     requirements=processing_requirements,
    #     system_site_packages=False,
    # )

    load_to_postgres_task = PythonVirtualenvOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres_venv,
        requirements=processing_requirements,
        system_site_packages=False,
    )

    end_task = EmptyOperator(task_id="end")

    # start_task >> [clean_and_process_fights_task, configure_fighters_dataset_task] \
    # >> configure_event_dataset_task >> load_to_postgres_task >> end_task

    start_task >> [clean_and_process_fights_task, configure_fighters_dataset_task] \
    >> load_to_postgres_task >> end_task

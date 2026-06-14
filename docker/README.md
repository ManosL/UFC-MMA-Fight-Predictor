# Infrastructure Docker Setup

## Overview

In this README, I will provide the necessary instructions to setup the infrastructure in order to run the ELT Pipeline, which runs the necessary scrapers, Python and SQL scripts and also the various downstreams of it.

## Requirements

- At least 10-12 GB of RAM
- Docker installed in your system
- PowerBI Desktop

## Spin Up the Docker Infrastructure

- Initially, do the initial Airflow setup by running the following command:
```
cd docker/
docker compose -f docker-compose.yml -f airflow-docker-compose.yaml up airflow-init
```
- After this finishes, setup the rest of the containers by running:
```
docker compose -f docker-compose.yml -f airflow-docker-compose.yaml up -d
```
- When the above command finishes, the following services will be started, in high level:
    - Airflow (the necessary services)
    - PostgreSQL (various instances)
    - PgAdmin
    - Minio
    - Scrapyd Server which spins up Scrapyd and ScrapydWeb
    - Python (one for data processing, one for Machine Learning)
    - MLflow
- You can access various apps from browser, through their corresponding UIs by accessing the following URLs and providing the creedentials as shown in the below table:

| Service Name  | URL                               | Username Env Variable         | Password Env Variable         |
|---------------|-----------------------------------|-------------------------------|-------------------------------|
| Airflow       | [localhost:8080](localhost:8080)  | `_AIRFLOW_WWW_USER_USERNAME`  | `_AIRFLOW_WWW_USER_PASSWORD`  |
| PgAdmin       | [localhost:80](localhost:80)      | `PGADMIN_EMAIL`               |  `PGADMIN_PASSWORD`           |
| MinIO         | [localhost:9000](localhost:9000)  | `MINIO_USERNAME`              | `MINIO_PASSWORD`              |
| Scrapyd       | [localhost:6800](localhost:6800)  | `SCRAPYD_USERNAME`            | `SCRAPYD_PASSWORD`            |
| ScrapydWeb    | [localhost:5000](localhost:5000)  | `SCRAPYD_USERNAME`            | `SCRAPYD_PASSWORD`            |
| MLflow        | [localhost:5100](localhost:5100)  | `MLFLOW_POSTGRES_USERNAME`    | `MLFLOW_POSTGRES_PASSWORD`    |

## Deploy the Spiders to Scrapyd

- Login to scrapyd-server's container:
```
docker exec -it docker-scrapy-server-1 bash
```
- Install the necessary dependencies:
```
pip install -r /requirements.txt
playwright install --with-deps
```
- Deploy the Spiders that are inside the `UFCStats_Crawlers` project:
```
cd /app/scrapyd/project/UFCStats_Crawlers
scrapyd-deploy -p UFCStats_Crawlers
```
- Logout from the container:
```
exit
```

Note: If you remove the containers or stop them, you should deploy again the Spiders by following the above steps.

## Clean up the containers, volumes and images

In order to remove the containers, the volumes and the downloaded images you should run the following:

```
docker compose -f docker-compose.yml -f airflow-docker-compose.yaml down --volumes --rmi all
```

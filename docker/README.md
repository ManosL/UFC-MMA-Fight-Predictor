# Infrastructure Docker Setup

## Overview

In this README, I will provide the necessary instructions to setup the infrastructure in order to run the ELT Pipeline, which runs the necessary scrapers, Python and SQL scripts and also the various downstreams of it.

## Requirements

- 8GB of RAM
- Docker installed in your system

## Spin Up the Docker Infrastructure

- Initially, do the initial Airflow setup by running the following command:
```
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
- You can access various apps from browser, through their corresponding UIs from the following URLs:
    - Airflow: [localhost:8080]
    - PgAdmin: [localhost:80]
    - Minio: [localhost:9000]
    - Scrapyd: [localhost:6800]
    - ScrapydWeb: [localhost:5000]
    - MLflow: [localhost:5100]

## Deploy the Spiders to Scrapyd

- Login to scrapyd-server's container:
```
docker exec -it docker-scrapy-server-1 bash
```
- Install the necessary dependencies:
```
pip install -r /requirements.txt
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

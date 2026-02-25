## Overview

In this README, I will provide the necessary instructions to run the Machine Learning experiments and the demo.

NOTE: This will be applicable until I create the ML Pipeline and the Application or the API that makes the predictions.

## Manual

### Installation instructions

If you've followed the instructions from [docker/README.md](https://github.com/ManosL/UFC-MMA-Fight-Predictor/blob/master/docker/README.md), you are ready to run the Experiments and the Demo.

### Experiments instructions

In order to run the experiments done in order to write the report, run the following command:

```
docker exec docker-machine-learning-python-1 python experiments.py
```

WARNING: This will take time in order to complete.

While running this program you will see logs in terminal and the resulting graphs will be inside the *src/machine_learning/results* directory.

### Demo instructions

In order to run the demo run the following command:
```
docker exec docker-machine-learning-python-1 python demo.py -p <prediction_dataset_path>
```
where the names of those variables are pretty descriptive, but we should also mention the following:

- Because the default dataset is in `src/data` folder the required command to run the demo is the following:
```
python demo.py -p ./data/Matchups.csv
```
I defined those parameters in case someone wants to run the demo with different datasets, but with the same
form.

- `<prediction_dataset_path>` is a csv file where each row has the following form:
```
<weight_class>|<title_fight(true or false)>|<rounds(3 or 5)>|<fighter_1_id>|<fighter_2_id>
```

- Demo trains a `RandomForestClassifier` with 150 estimators and the dataset is converted to be a
Double Difference dataset, because with these configurations I saw that out classifier gave the best
results. For more details, check `Machine_Learning_Report.pdf`.

- Note that because I could not take the gender of the fighters from their pages, I find initially
using a library, which is prone to error. Thus, there is a case that a correct matchup will raise
an error related to making matchup between fighters of different gender.

## Notes

- Data exist in `ML_Fighters` and `ML_Fights` views.
- To get the latest data you should run the `full_dag`DAG from [Airflow UI](localhost:8080).
- Match-ups should be created by hand in order to predict them.
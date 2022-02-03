#!/bin/bash

# Moving to spiders dir
cd ./datasets/UFCStats_Crawlers/UFCStats_Crawlers/

scrapy crawl event_spider
echo 'Finished retrieving event data'

scrapy crawl fighters_spider
echo 'Finished retrieving fighter data'

cd ../../

python3 Configure_Event_Dataset.py
echo 'Configure Final Fight Dataset'

python3 Configure_Fighters_Dataset.py
echo 'Configure Final Fighter Dataset'

rm UFC_Fighters.tsv
rm UFC_Fights_Train.tsv

echo 'Removed Middle Datasets'

mv Final_Fights_Dataset.csv ../src/data/Fights.csv
mv Final_Fighters_Dataset.csv ../src/data/Fighters.csv

echo 'Moved new datasets to ./src/data/'


cd ../
echo 'Now you have the most recent dataset'

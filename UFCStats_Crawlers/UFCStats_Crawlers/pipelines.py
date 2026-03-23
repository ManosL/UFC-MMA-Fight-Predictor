# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# TODO: BECAUSE THOSE TWO PIPELINES ARE SIMILAR EXPLOIT INHERITANCE
# TODO: ADD SOME STUFF AS ENVIRONMENT VARIABLES

import csv
import os
import sys

sys.path.append("/app/scrapyd/project/UFCStats_Crawlers/UFCStats_Crawlers/spiders")
sys.path.append("/app/common")

from minio_utils import MinioClient

DESTINATION_DIR = '/tmp'


class UFCFightStatsCrawlersPipeline:
    def open_spider(self, spider):
        if not os.path.exists(DESTINATION_DIR):
            os.mkdir(DESTINATION_DIR)

        self.minio_client = MinioClient(
            'minio:9000',
            access_key=os.environ.get('MINIO_USERNAME'),
            secret_key=os.environ.get('MINIO_PASSWORD'),
            secure=False
        )

        self.bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

        self.minio_client.create_bucket(self.bucket_name)

        common_columns = ['Fight_ID', 'Fight Date', 'Gender', 'Weight Class', 'Title Fight',
                          'Result', 'Method', 'Round', 'Time', 'Fight Time Format']

        fighter_1_prefix = 'Fighter 1 '
        fighter_2_prefix = 'Fighter 2 '

        fighter_columns = ['ID', 'Name', 'Nickname', 'Knock Downs', 'Sign.Strikes Done',
                           'Sign.Strikes Attempted', 'Sign.Strikes Perc.', 'Total Strikes Done',
		                    'Total Strikes Attempted', 'Takedowns Done', 'Takedowns Attempted',
                            'Takedowns Perc.', 'Submission Attempts', 'Rev.', 'Control Time']

        self.header_row = common_columns + [f'{fighter_1_prefix}{col}' for col in fighter_columns] \
                            + [f'{fighter_2_prefix}{col}' for col in fighter_columns]

        # TODO: IN FILE NAME BECAUSE MANY USERS MIGHT RUN AT THE SAME TIME APPEND A TIMESTAMP
        self.output_file_name = 'fight_new_actual_stats.csv'
        self.events_file_path = os.path.join(DESTINATION_DIR, self.output_file_name)
        self.events_file = open(self.events_file_path, 'w', newline='')
        self.event_writer = csv.writer(self.events_file, delimiter='|')

        self.event_writer.writerow(self.header_row)
        return



    def process_item(self, item, spider):
        row_to_add = [item[col] for col in self.header_row]

        self.event_writer.writerow(row_to_add)
        self.events_file.flush()

        return item



    def close_spider(self, spider):
        self.events_file.close()

        self.minio_client.write_file(
            self.bucket_name,
            self.events_file_path,
            self.output_file_name
        )

        os.remove(self.events_file_path)
        return



class UFCFightersGeneralStatsCrawlersPipeline:
    def open_spider(self, spider):
        if not os.path.exists(DESTINATION_DIR):
            os.mkdir(DESTINATION_DIR)

        self.minio_client = MinioClient(
            'minio:9000',
            access_key=os.environ.get('MINIO_USERNAME'),
            secret_key=os.environ.get('MINIO_PASSWORD'),
            secure=False
        )

        self.bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

        self.minio_client.create_bucket(self.bucket_name)

        self.header_row = ['Fighter ID', 'Fighter Name', 'Wins', 'Loses',
                           'Draws', 'Height', 'Weight', 'Reach', 'Stance',
                           'DOB', 'SLpM', 'Str.Acc.', 'SApM', 'Str. Def.',
                           'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']

        self.output_file_name = 'fighters_new_current_stats.csv'
        self.fighters_file_path = os.path.join(DESTINATION_DIR, self.output_file_name)
        self.fighters_file = open(self.fighters_file_path, 'w', newline='')
        self.fighters_writer = csv.writer(self.fighters_file, delimiter='|')

        self.fighters_writer.writerow(self.header_row)
        return



    def process_item(self, item, spider):
        row_to_add = [item[col] for col in self.header_row]

        self.fighters_writer.writerow(row_to_add)
        self.fighters_file.flush()

        return item



    def close_spider(self, spider):
        self.fighters_file.close()

        self.minio_client.write_file(
            self.bucket_name,
            self.fighters_file_path,
            self.output_file_name
        )

        os.remove(self.fighters_file_path)
        return

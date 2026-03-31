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

import csv
import os

from common.schemas import CRAWLED_FIGHTER_COLUMNS, get_crawled_fight_columns
from common.minio_utils import MinioClient

DESTINATION_DIR = '/tmp'


class BaseCSVPipeline:
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

        # TODO: IN FILE NAME BECAUSE MANY USERS MIGHT RUN AT THE SAME TIME APPEND A TIMESTAMP
        self.csv_file_path = os.path.join(DESTINATION_DIR, self.output_file_name)
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter='|')

        self.csv_writer.writerow(self.header_row)
        return


    def process_item(self, item, spider):
        row_to_add = [item[col] for col in self.header_row]

        self.csv_writer.writerow(row_to_add)
        self.csv_file.flush()

        return item


    def close_spider(self, spider):
        self.csv_file.close()

        self.minio_client.write_file(
            self.bucket_name,
            self.csv_file_path,
            self.output_file_name
        )

        os.remove(self.csv_file_path)
        return
    

class UFCFightStatsCrawlersPipeline(BaseCSVPipeline):
    header_row: list[str] = get_crawled_fight_columns()
    output_file_name: str = 'fight_new_actual_stats.csv'


class UFCFightersGeneralStatsCrawlersPipeline(BaseCSVPipeline):
    header_row: list[str] = CRAWLED_FIGHTER_COLUMNS
    output_file_name: str = 'fighters_new_current_stats.csv'

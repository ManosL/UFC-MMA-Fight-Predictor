# TODO: EXPLOIT UTILS FOR THE MINIO STUFF THIS APPLIES EVERYWHERE
import sys
from copy import deepcopy
import os
import pandas as pd
import string
import scrapy
import re
from minio import Minio
from io import BytesIO

sys.path.append("/app/scrapyd/project/UFCStats_Crawlers/UFCStats_Crawlers/spiders")

from utils import log_path_to_scrapyd_url, get_minio_client
from utils import create_minio_bucket, write_json_to_minio


class FightersSpider(scrapy.Spider):
    name = 'fighters_spider'
    start_urls = ['http://www.ufcstats.com/statistics/events/completed']

    custom_settings = {
        "ITEM_PIPELINES": {
            "UFCStats_Crawlers.pipelines.UFCFightersGeneralStatsCrawlersPipeline": 300,
        }
    }

    ######################## HELPER METHODS ##############################
    def get_fighter_id_from_url(self, url):
        id = re.match(r'^.*ufcstats.com/fighter-details/([a-zA-Z0-9]*)(\/|\?)?.*$', url).groups()[0]
        assert(id != None)

        return id
    ######################################################################


    def parse(self, response):
        self.is_incremental = False if self.is_incremental.lower() == "false" else True
        self.logger.debug(f'Will spider run incrementally: {self.is_incremental}')

        self.attrs = ['Fighter ID', 'Fighter Name', 'Wins', 'Loses', 'Draws', 'Height',
                    'Weight', 'Reach', 'Stance', 'DOB', 'SLpM', 'Str.Acc.', 'SApM',
                    'Str. Def.', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']

        if self.is_incremental:
            minio_client = Minio(
                'minio:9000',
                access_key=os.environ.get('MINIO_USERNAME'),
                secret_key=os.environ.get('MINIO_PASSWORD'),
                secure=False
            )

            response = minio_client.get_object(
                os.environ.get("MINIO_RAW_DATA_BUCKET_NAME"),
                "fight_new_actual_stats.csv"
            )

            event_data = pd.read_csv(BytesIO(response.read()), sep='|', header=0)

            response.close()
            response.release_conn()

            all_fighter_ids = set(event_data['Fighter 1 ID']).union(set(event_data['Fighter 2 ID']))
            self.logger.debug(f"Will crawl info for the following Fighter IDs: {all_fighter_ids}")

            self.crawler.stats.inc_value("fighters_expected", len(all_fighter_ids))
            for fighter_id in all_fighter_ids:
                next_url = f"http://ufcstats.com/fighter-details/{fighter_id}"

                yield scrapy.Request(url=next_url, callback=self.fighter_parse)
        else:
            links = ['http://www.ufcstats.com/statistics/fighters?char=' + l + '&page=all'
                        for l in list(string.ascii_lowercase)]

            for link in links:
                yield scrapy.Request(url=link, callback=self.letter_parse)

    def letter_parse(self, response):
        links = []

        table_rows = response.css('table.b-statistics__table tbody tr.b-statistics__table-row')
        table_rows = table_rows[1:]

        for row in table_rows:
            col = row.css('td.b-statistics__table-col')[0]
            links.append(col.css('a::attr(href)').get())

        print(len(links),response.url)
        self.crawler.stats.inc_value("fighters_expected", len(links))
        for link in links:
            yield scrapy.Request(url = link,callback = self.fighter_parse)


    def fighter_parse(self, response):
        final_row = []

        # Getting fighter's ID
        fighter_id = self.get_fighter_id_from_url(response.url)
        final_row.append(fighter_id)

        # Getting fighter's name
        pg_title = response.css('h2.b-content__title')

        final_row.append(pg_title.css('span.b-content__title-highlight::text').get())

        final_row[1] = re.sub('\n','',final_row[1])
        final_row[1] = final_row[1].split()
        final_row[1] = ' '.join(list(filter(lambda x: x != '',final_row[1])))

        # Getting fighter's record

        record = pg_title.css('span.b-content__title-record::text').get()
        record = re.sub('Record: ','',record)
        record = re.sub('\n','',record)
        record = re.sub(' ','',record)
        record = re.sub(r'\(.*?\)','',record)
        record = record.split('-')   # getting a list of form [win,lose,draw]
        record = [int(x) for x in record]

        final_row = final_row + record

        # Details table
        details_table = response.css('div.b-fight-details.b-fight-details_margin-top')

        # Getting first sub-table
        curr_table = details_table.css('div.b-list__info-box.' +
                            'b-list__info-box_style_small-width.js-guide ' +
                            'ul.b-list__box-list')

        curr_table = curr_table.css('li.b-list__box-list-item.' +
                                'b-list__box-list-item_type_block').getall()


        for elem in curr_table:
            elem = re.sub('\n','',elem)
            elem = re.match('<li (.*?)>(.*)</li>',elem).groups()[1]
            elem = re.sub('<i (.*)>(.*)</i>','',elem)
            elem = ' '.join(list(filter(lambda x: x != '',elem.split())))
            if elem == '' or elem == '--':
                elem = 'No Stat'

            final_row.append(elem)

        # Getting the second sub-table(Career Statistics)
        curr_table = details_table.css('div.b-list__info-box.b-list__info' +
                                '-box_style_middle-width.js-guide.clearfix ' +
                                'div.b-list__info-box-left.clearfix')

        curr_sub_table = curr_table.css('ul.b-list__box-list.' +
                                            'b-list__box-list_margin-top ' +
                                            'li.b-list__box-list-item.' +
                                            'b-list__box-list-item_type_block').getall()

        for elem in curr_sub_table:
            elem = re.sub('\n','',elem)
            elem = re.match('<li (.*?)>(.*)</li>',elem).groups()[1]
            elem = re.sub('<i (.*)>(.*)</i>','',elem)
            elem = ' '.join(list(filter(lambda x: x != '',elem.split())))
            if elem != '' and elem != '--':
                final_row.append(elem)

        curr_item = {col: val for col, val in zip(self.attrs, final_row)}

        self.crawler.stats.inc_value("fighters_parsed", 1)
        yield curr_item


    def closed(self, reason):
        minio_client = get_minio_client()
        log_file_url = log_path_to_scrapyd_url(self.settings["LOG_FILE"])

        self.crawler.stats.set_value("spider_name", self.name)
        self.crawler.stats.set_value("log_file_url", log_file_url)

        stats_file_name = os.path.basename(self.settings["LOG_FILE"])

        bucket_name = os.environ.get('MINIO_FIGHTER_CRAWL_LOGS_BUCKET_NAME')

        self.logger.info(create_minio_bucket(minio_client, bucket_name))

        crawl_stats = deepcopy(self.crawler.stats.get_stats())
        crawl_stats["start_time"] = str(crawl_stats["start_time"])

        write_json_to_minio(
            minio_client,
            bucket_name,
            stats_file_name,
            crawl_stats
        )

        return

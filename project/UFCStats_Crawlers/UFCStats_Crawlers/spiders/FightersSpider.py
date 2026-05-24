import os
import string
import scrapy

from common.minio_utils import MinioClient
from common.schemas import CRAWLED_FIGHTER_COLUMNS

from UFCStats_Crawlers.spiders.parsers.fighter_parsers import parse_fighter_name, parse_fighter_record
from UFCStats_Crawlers.spiders.parsers.fighter_parsers import parse_measurements_table, parse_career_stats_table

from UFCStats_Crawlers.spiders.utils import get_fighter_id_from_url
from UFCStats_Crawlers.spiders.utils import save_crawling_stats_to_minio


class FightersSpider(scrapy.Spider):
    name = 'fighters_spider'
    start_urls = ['http://www.ufcstats.com/statistics/events/completed']

    custom_settings = {
        "ITEM_PIPELINES": {
            "UFCStats_Crawlers.pipelines.UFCFightersGeneralStatsCrawlersPipeline": 300,
        }
    }

    def parse(self, response):
        self.is_incremental = False if self.is_incremental.lower() == "false" else True
        self.logger.debug(f'Will spider run incrementally: {self.is_incremental}')

        if self.is_incremental:
            minio_client = MinioClient(
                'minio:9000',
                access_key=os.environ.get('MINIO_USERNAME'),
                secret_key=os.environ.get('MINIO_PASSWORD'),
                secure=False
            )

            job_id = os.path.basename(self.settings["LOG_FILE"]).split('.')[0]

            event_data = minio_client.read_csv_to_pandas(
                os.environ.get("MINIO_RAW_DATA_BUCKET_NAME"),
                os.path.join(job_id, "fight_new_actual_stats.csv"),
                sep='|', 
                header=0
            )

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

        print(len(links), response.url)
        self.crawler.stats.inc_value("fighters_expected", len(links))
        for link in links:
            yield scrapy.Request(url=link, callback=self.fighter_parse)


    def fighter_parse(self, response):
        final_row = []

        fighter_id = get_fighter_id_from_url(response.url)
        final_row.append(fighter_id)

        pg_title = response.css('h2.b-content__title')
        name_div = pg_title.css('span.b-content__title-highlight::text').get()
        
        fighter_name = parse_fighter_name(name_div)

        final_row.append(fighter_name)

        record_div = pg_title.css('span.b-content__title-record::text').get()

        final_row.extend(parse_fighter_record(record_div))

        details_table = response.css('div.b-fight-details.b-fight-details_margin-top')

        measurements_table = details_table.css('div.b-list__info-box.' +
                                'b-list__info-box_style_small-width.js-guide ' +
                                'ul.b-list__box-list')

        measurements_table = measurements_table.css('li.b-list__box-list-item.' +
                                        'b-list__box-list-item_type_block').getall()

        final_row.extend(list(parse_measurements_table(measurements_table)))

        career_stats_table = details_table.css('div.b-list__info-box.b-list__info' +
                                '-box_style_middle-width.js-guide.clearfix ' +
                                'div.b-list__info-box-left.clearfix')

        career_stats_table = career_stats_table.css('ul.b-list__box-list.' +
                                            'b-list__box-list_margin-top ' +
                                            'li.b-list__box-list-item.' +
                                            'b-list__box-list-item_type_block').getall()

        final_row.extend(list(parse_career_stats_table(career_stats_table)))

        curr_item = {col: val for col, val in zip(CRAWLED_FIGHTER_COLUMNS, final_row)}

        self.crawler.stats.inc_value("fighters_parsed", 1)
        yield curr_item


    def closed(self, reason):
        bucket_name = os.environ.get('MINIO_FIGHTER_CRAWL_LOGS_BUCKET_NAME')

        save_crawling_stats_to_minio(self, bucket_name)

        return

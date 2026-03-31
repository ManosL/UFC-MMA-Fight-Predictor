import sys
from datetime import datetime
import scrapy
import re
import os

from common.schemas import get_crawled_fight_columns

from UFCStats_Crawlers.spiders.parsers.fight_parsers import process_event_name, process_bout_description
from UFCStats_Crawlers.spiders.parsers.fight_parsers import get_fighter_id_name_and_nickname
from UFCStats_Crawlers.spiders.parsers.fight_parsers import get_fight_result, parse_total_stats_table
from UFCStats_Crawlers.spiders.parsers.fight_parsers import parse_outcome_section

from UFCStats_Crawlers.spiders.utils import get_fight_id_from_url, save_crawling_stats_to_minio


class EventSpider(scrapy.Spider):
	name = 'event_spider'
	start_urls = ['http://www.ufcstats.com/statistics/events/completed?page=all']

	custom_settings = {
        "ITEM_PIPELINES": {
            "UFCStats_Crawlers.pipelines.UFCFightStatsCrawlersPipeline": 300,
        }
    }

	def parse(self, response):
		self.is_incremental = False if self.is_incremental.lower() == "false" else True

		self.logger.debug(f'Will spider run incrementally: {bool(self.is_incremental)}')
		self.logger.debug(f'Lookup days are: {int(self.lookup_days)} days')

		self.event_columns = get_crawled_fight_columns()

		events_array_selector = 'tr.b-statistics__table-row'

		rows = response.css(events_array_selector)[2:]
		event_links = []
		event_dates = []

		# Getting the href of each event
		for row in rows:
			event_link_selector = 'td.b-statistics__table-col i.b-statistics__table-content a::attr(href)'

			event_date_selector = '''td.b-statistics__table-col i.b-statistics__table-content
									span.b-statistics__date::text'''

			event_date = row.css(event_date_selector).get().strip()
			days_since_happened = datetime.today().date() - datetime.strptime(event_date, "%B %d, %Y").date()

			if not self.is_incremental or (days_since_happened.days <= int(self.lookup_days)):
				event_link = row.css(event_link_selector).get()
				event_links.append(event_link)

				event_dates.append(event_date)
			else:
				break

		self.crawler.stats.inc_value("events_expected", len(event_links))

		for link, date in zip(event_links, event_dates, strict=True):
			yield scrapy.Request(url=link, callback=self.event_parse, meta={'event_date': date})



	def event_parse(self, response):
		event_name_selector = 'div.l-page__container h2.b-content__title span::text'

		event_name = response.css(event_name_selector).get()
		event_name = process_event_name(event_name)

		event_matches_selector = 	'table.b-fight-details__table.b-fight-' \
									'details__table_style_margin-top.b-fight' \
									'-details__table_type_event' \
									'-details.js-fight-table'

		rows = response.css(event_matches_selector)
		rows = rows.css('tr::attr(data-link)').getall()

		self.crawler.stats.inc_value("events_parsed", 1)
		self.crawler.stats.inc_value("fights_expected", len(rows))

		for link in rows:
			yield scrapy.Request(url = link, callback=self.fight_parse,
								meta={
									'event_date': response.meta.get('event_date'),
									'fight_id': get_fight_id_from_url(link)
								})


	def fight_parse(self,response):
		fighter1_info = []
		fighter2_info = []
		fight_info = []

		event_name_selector = 'h2.b-content__title a::text'
		event_name = response.css(event_name_selector).get()

		if event_name is None:
			self.logger.warning(f"Error at {response.url}")
			return None

		event_name = process_event_name(event_name)

		# Retrieving the weight class + other info about the fight.
		# Generally this is something like "Lightweight Bout" or
		# "Women's Strawweight Title Fight", etc. so we can extract
		# Gender, Weight Class and whether its a title bout.
		bout_desc_selector = '''div.b-fight-details__fight
								div.b-fight-details__fight-head
								i.b-fight-details__fight-title'''

		bout_desc = response.css(bout_desc_selector).get()
		
		fight_id = response.meta.get('fight_id')
		event_date = response.meta.get('event_date')
		gender, title_fight, weight_class = process_bout_description(bout_desc)

		fight_info.extend([fight_id, event_date, gender, weight_class, title_fight])

		fighters = response.css('div.b-fight-details__persons.clearfix')
		fighters = fighters.css('div.b-fight-details__person')

		# The result will be written by the perspective of the
		# fighter that is written firstly.
		result = get_fight_result(fighters[0])

		fight_info += [result]

		fighter1 = fighters[0]
		fighter2 = fighters[1]

		fighter1_info.extend(get_fighter_id_name_and_nickname(fighter1))
		fighter2_info.extend(get_fighter_id_name_and_nickname(fighter2))

		# Parses fight info [result,method,round,time,match round mode]

		fight = response.css('''div.b-fight-details__fight div.b-fight-details__content
							p.b-fight-details__text''')[0]

		fight_info.extend(parse_outcome_section(fight))

		# Parses fight stats for both fighters
		match_stats_table = response.css('tbody.b-fight-details__table-body')

		if match_stats_table == []:
			fighter1_info.extend(['No Stats'] * 12)
			fighter2_info.extend(['No Stats'] * 12)
		else:
			match_stats_table = match_stats_table[0]  # It's the first on the webpage

			fighter_1_totals, fighter_2_totals = parse_total_stats_table(match_stats_table)
			
			fighter1_info.extend(fighter_1_totals)
			fighter2_info.extend(fighter_2_totals)

		final_attrs = fight_info + fighter1_info + fighter2_info

		curr_item = {col: val for col, val in zip(self.event_columns, final_attrs)}

		self.crawler.stats.inc_value("fights_parsed", 1)
		yield curr_item


	def closed(self, reason):
		bucket_name = os.environ.get('MINIO_EVENT_CRAWL_LOGS_BUCKET_NAME')

		save_crawling_stats_to_minio(self, bucket_name)

		return

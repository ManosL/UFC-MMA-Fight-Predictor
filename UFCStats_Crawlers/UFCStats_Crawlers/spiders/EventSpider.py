import sys
from copy import deepcopy
from datetime import datetime
import scrapy
import re
import os

sys.path.append("/app/scrapyd/project/UFCStats_Crawlers/UFCStats_Crawlers/spiders")

from utils import log_path_to_scrapyd_url, get_minio_client
from utils import create_minio_bucket, write_json_to_minio


# these weight classes are taken from https://en.wikipedia.org/wiki/Mixed_martial_arts_weight_classes
mma_weight_classes = ['atomweight', 'strawweight', 'flyweight', 'bantamweight', 'featherweight', 'lightweight',
					'super lightweight', 'welterweight', 'super welterweight', 'middleweight', 'super middleweight',
					'light heavyweight', 'cruiserweight', 'heavyweight']

class EventSpider(scrapy.Spider):
	name = 'event_spider'
	start_urls = ['http://www.ufcstats.com/statistics/events/completed?page=all']

	custom_settings = {
        "ITEM_PIPELINES": {
            "UFCStats_Crawlers.pipelines.UFCFightStatsCrawlersPipeline": 300,
        }
    }

	##################### Web-Page parsing helpers functions ##################

	def process_event_name(self, event_name):
		# Removing leading and trailing whitespace
		event_name = event_name.strip()

		return event_name


	def process_bout_description(self, bout_desc):
		# Check how you get the weight class because there are other keywords used
		#print(bout_desc)
		bout_desc = re.match(r'<i.*>(?:.*<img.*>)?(.*)</i>', bout_desc, flags=re.DOTALL).groups()[0]
		bout_desc = bout_desc.strip()

		fight_weight_class = 'catch weight'

		# Searching the weight class in the description according to known MMA weight classes
		for weight_class in mma_weight_classes:
			if re.search(weight_class, bout_desc, flags=re.IGNORECASE):
				fight_weight_class = weight_class.lower()
				break

		if re.search(r'Title|Interim', bout_desc, flags=re.IGNORECASE):
			title_fight = True
		else:
			title_fight = False

		if re.search('Women', bout_desc, flags=re.IGNORECASE):
			gender = 'female'
		else:
			gender = 'male'

		#print("TLF " + str(title_fight) + " WEIGHT CLASS " + weight_class)
		return gender, title_fight, fight_weight_class


	def get_fighter_id_from_url(self, url):
		id = re.match(r'^.*ufcstats.com/fighter-details/([a-zA-Z0-9]*)(\/|\?)?.*$', url).groups()[0]
		assert(id != None)

		return id.strip()


	def get_fighter_id_name_and_nickname(self, fighter_html):
		fighter_id = fighter_html.css('''div.b-fight-details__person-text
									h3.b-fight-details__person-name
									a::attr(href)''').get().strip()

		if fighter_id is None:
			fighter_id = fighter_html.css('''div.b-fight-details__person-text
										h3.b-fight-details__person-name
										span.b-link.b-fight-details__person-link::attr(href)''').get().strip()

		fighter_id = self.get_fighter_id_from_url(fighter_id)

		fighter_name = fighter_html.css('''div.b-fight-details__person-text
									h3.b-fight-details__person-name
									a::text''').get().strip()

		if fighter_name is None:
			fighter_name = fighter_html.css('''div.b-fight-details__person-text
										h3.b-fight-details__person-name
										span.b-link.b-fight-details__person-link::text''').get().strip()

		fighter_nickname = fighter_html.css('''div.b-fight-details__person-text
					p.b-fight-details__person-title::text''').get()
		fighter_nickname = re.sub('\n','', fighter_nickname)
		fighter_nickname = re.match(r'(.*)"(.*)"', fighter_nickname)

		if fighter_nickname is None:
			fighter_nickname = 'No_Nickname'
		else:
			fighter_nickname = fighter_nickname.groups()[1].strip()

		return fighter_id, fighter_name, fighter_nickname


	def get_fight_result(self, fighter1_html, url=''):
		result = fighter1_html.css('i.b-fight-details__person-status.b-fight-' +
								'details__person-status_style_green::text').get()
		if result != None:
			result = re.sub(r'\W+','', result)

		if result == 'W':
			result = 'win'
		else:
			result = fighter1_html.css('i.b-fight-details__person-status.b' +
									'-fight-details__person-status_style_gray::text').get()

			result = re.sub('[^a-zA-Z0-9_]+','', result)
			if result == 'L':
				result = 'lose'
			elif result == 'D':
				result = 'draw'
			elif result == 'NC':
				result = 'no contest'
			else:
				result = 'error'

		return result


	##################### Web-Page parsing functions ##########################
	def parse(self, response):
		self.is_incremental = False if self.is_incremental.lower() == "false" else True

		self.logger.debug(f'Will spider run incrementally: {bool(self.is_incremental)}')
		self.logger.debug(f'Lookup days are: {int(self.lookup_days)} days')

		common_columns = ['Fight Date', 'Gender', 'Weight Class', 'Title Fight',
						  'Result', 'Method', 'Round', 'Time', 'Fight Time Format']

		fighter_1_prefix = 'Fighter 1 '
		fighter_2_prefix = 'Fighter 2 '

		fighter_columns = ['ID', 'Name', 'Nickname', 'Knock Downs', 'Sign.Strikes Done',
                       	   'Sign.Strikes Attempted', 'Sign.Strikes Perc.', 'Total Strikes Done',
		                   'Total Strikes Attempted', 'Takedowns Done', 'Takedowns Attempted',
                           'Takedowns Perc.', 'Submission Attempts', 'Rev.', 'Control Time']

		self.event_columns = common_columns + [f'{fighter_1_prefix}{col}' for col in fighter_columns] \
                            + [f'{fighter_2_prefix}{col}' for col in fighter_columns]

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
				event_links = [event_link] + event_links # TODO: WHY I ADD THIS TO THE START?

				event_dates = [event_date] + event_dates
			else:
				break

		self.crawler.stats.inc_value("events_expected", len(event_links))

		for link, date in zip(event_links, event_dates, strict=True):
			yield scrapy.Request(url=link, callback=self.event_parse, meta={'event_date': date})



	def event_parse(self, response):
		event_name_selector = 'div.l-page__container h2.b-content__title span::text'

		event_name = response.css(event_name_selector).get()
		event_name = self.process_event_name(event_name)

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
								meta={'event_date': response.meta.get('event_date')})


	def fight_parse(self,response):
		fighter1_info = []
		fighter2_info = []
		fight_info = []

		event_name_selector = 'h2.b-content__title a::text'
		event_name = response.css(event_name_selector).get()

		if event_name is None:
			self.logger.warning(f"Error at {response.url}")
			return None

		event_name = self.process_event_name(event_name)

		# Retrieving the weight class + other info about the fight.
		# Generally this is something like "Lightweight Bout" or
		# "Women's Strawweight Title Fight", etc. so we can extract
		# Gender, Weight Class and whether its a title bout.
		bout_desc_selector = '''div.b-fight-details__fight
								div.b-fight-details__fight-head
								i.b-fight-details__fight-title'''

		bout_desc = response.css(bout_desc_selector).get()

		event_date = response.meta.get('event_date')
		gender, title_fight, weight_class = self.process_bout_description(bout_desc)

		fight_info = [event_date, gender, weight_class, title_fight]

		fighters = response.css('div.b-fight-details__persons.clearfix')
		fighters = fighters.css('div.b-fight-details__person')

		# The result will be written by the perspective of the
		# fighter that is written firstly.
		result = self.get_fight_result(fighters[0], response.request.url)

		fight_info += [result]

		fighter1 = fighters[0]
		fighter2 = fighters[1]

		# Get fighter ID, the full name and nickname
		fighter1_id, fighter1_name, fighter1_nickname = \
			self.get_fighter_id_name_and_nickname(fighter1)

		fighter1_info += [fighter1_id, fighter1_name, fighter1_nickname]

		fighter2_id, fighter2_name, fighter2_nickname = \
			self.get_fighter_id_name_and_nickname(fighter2)

		fighter2_info += [fighter2_id, fighter2_name, fighter2_nickname]

		# Gives to fight info [result,method,round,time,match round mode]

		fight = response.css('''div.b-fight-details__fight div.b-fight-details__content
							p.b-fight-details__text''')[0]

		fight_info.append(fight.css('i.b-fight-details__text-item_first i::text').getall()[1].strip())

		items = fight.css('i.b-fight-details__text-item').getall()[0:3]

		for item in items:
			item = re.sub('\n','',item)
			item = re.sub(r'<i class="b-fight-details__label">.*?</i>', '', item)
			item = re.match(r'<i class="b-fight-details__text-item">(.*)</i>',item).groups()[0]
			item = re.sub(' ','',item)

			fight_info.append(item)

		match_stats_table = response.css('tbody.b-fight-details__table-body')

		if match_stats_table == []:
			fighter1_info += ['No Stats'] * 12
			fighter2_info += ['No Stats'] * 12
		else:
			match_stats_table = match_stats_table[0]  # It's the first on the webpage

			match_stats_table = match_stats_table.css('''tr.b-fight-details__table-row
							td.b-fight-details__table-col''')[1:]

			for stat in match_stats_table:
				stat = stat.css('p.b-fight-details__table-text::text').getall()

				stat1 = re.sub(r'\s+','',stat[0])
				stat2 = re.sub(r'\s+','',stat[1])

				stat1_groups = re.match(r'(.*)of(.*)',stat1)
				stat2_groups = re.match(r'(.*)of(.*)',stat2)

				if stat1_groups is not None:
					stat1_groups = stat1_groups.groups()
					stat2_groups = stat2_groups.groups()

					fighter1_info += stat1_groups
					fighter2_info += stat2_groups
				else:
					fighter1_info.append(stat1)
					fighter2_info.append(stat2)

		final_attrs = fight_info + fighter1_info + fighter2_info

		curr_item = {col: val for col, val in zip(self.event_columns, final_attrs)}

		self.crawler.stats.inc_value("fights_parsed", 1)
		yield curr_item


	def closed(self, reason):
		minio_client = get_minio_client()
		log_file_url = log_path_to_scrapyd_url(self.settings["LOG_FILE"])

		self.crawler.stats.set_value("spider_name", self.name)
		self.crawler.stats.set_value("log_file_url", log_file_url)

		stats_file_name = os.path.basename(self.settings["LOG_FILE"])

		bucket_name = os.environ.get('MINIO_EVENT_CRAWL_LOGS_BUCKET_NAME')

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

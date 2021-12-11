import scrapy
import re
import csv

# these weight classes are taken from https://en.wikipedia.org/wiki/Mixed_martial_arts_weight_classes
mma_weight_classes = ['atomweight', 'strawweight', 'flyweight', 'bantamweight', 'featherweight', 'lightweight',
					'super lightweight', 'welterweight', 'super welterweight', 'middleweight', 'super middleweight',
					'light heavyweight', 'cruiserweight', 'heavyweight']

events_dates = {}

class EventSpider(scrapy.Spider):
	name = 'event_spider'
	start_urls = ['http://www.ufcstats.com/statistics/events/completed?page=all']

	##################### Web-Page parsing helpers functions ##################

	def process_event_name(self, event_name):
		# Removing leading and trailing whitespace
		event_name = event_name.strip()

		return event_name

	def process_event_date(self, event_date):
		event_date = re.match('<li.*>.*<i.*>.*</i>(.*)</li>', event_date, flags=re.DOTALL).groups()[0]
		event_date = event_date.strip()
		return event_date

	def process_bout_description(self, bout_desc):
		# Check how you get the weight class because there are other keywords used
		#print(bout_desc)
		bout_desc = re.match('<i.*>(?:.*<img.*>)?(.*)</i>', bout_desc, flags=re.DOTALL).groups()[0]
		bout_desc = bout_desc.strip()

		fight_weight_class = 'catch weight'

		# Searching the weight class in the description according to known MMA weight classes
		for weight_class in mma_weight_classes:
			if re.search(weight_class, bout_desc, flags=re.IGNORECASE):
				fight_weight_class = weight_class.lower()
				break

		if re.search('Title|Interim', bout_desc, flags=re.IGNORECASE):
			title_fight = True
		else:
			title_fight = False

		if re.search('Women', bout_desc, flags=re.IGNORECASE):
			gender = 'female'
		else:
			gender = 'male'

		#print("TLF " + str(title_fight) + " WEIGHT CLASS " + weight_class)
		return gender, title_fight, fight_weight_class

	##################### Web-Page parsing functions ##########################
	def parse(self,response):
		rows = response.css('tr.b-statistics__table-row')[2:]
		event_links = []

		for row in rows:
			event_link = row.css('td.b-statistics__table-col i.b-statistics__table-content a::attr(href)').get()
			event_links = [event_link] + event_links

		#print(event_links)
		for link in event_links:
			yield scrapy.Request(url = link, callback = self.event_parse)

	def event_parse(self,response):
		event_name = response.css('div.l-page__container h2.b-content__title span::text').get()
		event_name = self.process_event_name(event_name)

		event_date = response.css('''div.l-page__container div.b-fight-details
							div.b-list__info-box.b-list__info-box_style_large-width
							ul.b-list__box-list li.b-list__box-list-item''').get()
		
		event_date = self.process_event_date(event_date)

		events_dates[event_name] = event_date

		rows = response.css('table.b-fight-details__table.b-fight-details__table_style_margin-top.b-fight-details__table_type_event-details.js-fight-table')
		rows = rows.css('tr::attr(data-link)').getall()

		for link in rows:
			yield scrapy.Request(url = link, callback=self.fight_parse)

	def fight_parse(self,response):
		fighter1_info = []
		fighter2_info = []
		fight_info = []

		event_name = response.css('h2.b-content__title a::text').get()
		if event_name is None:
			with open('mymflog.txt','a') as log:
				log.write('Error at ' + response.url + '\n')
				return 0
		
		event_name = self.process_event_name(event_name)

		# Retrieving the weight class
		bout_desc = response.css('''div.b-fight-details__fight div.b-fight-details__fight-head
							i.b-fight-details__fight-title''').get()

		gender, title_fight, weight_class = self.process_bout_description(bout_desc)

		fight_info = [events_dates[event_name], gender, weight_class, title_fight]

		fighters = response.css('div.b-fight-details__persons.clearfix')
		fighters = fighters.css('div.b-fight-details__person')

		# The result will be written by first fighter's written 
		# perspective

		result = fighters[0].css('i.b-fight-details__person-status.b-fight-' +
								'details__person-status_style_green::text').get()
		if result != None:
			result = re.sub('\W+','', result )

		if result == 'W':
			result = 'win'
		else:
			result = fighters[0].css('i.b-fight-details__person-status.b' +
									'-fight-details__person-status_style_gray::text').get()
			
			result = re.sub('[^a-zA-Z0-9_]+','', result )
			if result == 'L':
				result = 'lose'
			elif result == 'D':
				result = 'draw'
			elif result == 'NC':
				result = 'no contest'
			else:
				f = open('mylog.txt','a')
				f.write('found error ' + response.request.url + '\n')
				result = 'error'

		fight_info += [result]

		fighter1 = fighters[0]
		fighter2 = fighters[1]

		# Gives fighter the full name and nickname
		fighter1_info.append(fighter1.css('''div.b-fight-details__person-text 
					h3.b-fight-details__person-name
					a::text''').get().strip())
		
		if fighter1_info[0] is None:
			fighter1_info = [
					fighter1.css('''div.b-fight-details__person-text 
					h3.b-fight-details__person-name
					span.b-link.b-fight-details__person-link::text''').get().strip()
			]

		fighter1_nickname = fighter1.css('''div.b-fight-details__person-text 
					p.b-fight-details__person-title::text''').get()
		fighter1_nickname = re.sub('\n','',fighter1_nickname)
		fighter1_nickname = re.match(r'(.*)"(.*)"',fighter1_nickname)

		if fighter1_nickname is None:
			fighter1_info.append('No_Nick')
		else:
			fighter1_info.append(fighter1_nickname.groups()[1].strip())

		fighter2_info.append(fighter2.css('''div.b-fight-details__person-text 
					h3.b-fight-details__person-name
					a::text''').get().strip())

		if fighter2_info[0] is None:
			fighter2_info = [
					fighter2.css('''div.b-fight-details__person-text 
					h3.b-fight-details__person-name
					span::text''').get().strip()
			]

		fighter2_nickname = fighter2.css('''div.b-fight-details__person-text 
					p.b-fight-details__person-title::text''').get()
		fighter2_nickname = re.sub('\n','',fighter2_nickname)		
		fighter2_nickname = re.match(r'(.*)"(.*)"',fighter2_nickname)

		if fighter2_nickname is None:
			fighter2_info.append('No_Nickname')
		else:
			fighter2_info.append(fighter2_nickname.groups()[1])

		fight = response.css('''div.b-fight-details__fight div.b-fight-details__content
							p.b-fight-details__text''')[0]

		# Gives to fight info [result,method,round,time,match round mode]
		fight_info.append(fight.css('i.b-fight-details__text-item_first i::text').getall()[1])

		items = fight.css('i.b-fight-details__text-item').getall()[0:3]

		for item in items:
			item = re.sub('\n','',item)
			item = re.sub(r'<i class="b-fight-details__label">.*?</i>', '', item)
			item = re.match('<i class="b-fight-details__text-item">(.*)</i>',item).groups()[0]
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

				stat1 = re.sub('\s+','',stat[0])
				stat2 = re.sub('\s+','',stat[1])

				stat1_groups = re.match('(.*)of(.*)',stat1)
				stat2_groups = re.match('(.*)of(.*)',stat2)
				
				if stat1_groups is not None:
					stat1_groups = stat1_groups.groups()
					stat2_groups = stat2_groups.groups()

					fighter1_info += stat1_groups
					fighter2_info += stat2_groups
				else:
					fighter1_info.append(stat1)
					fighter2_info.append(stat2)

		final = fight_info + fighter1_info + fighter2_info

		""" WRITING TO tsv FILE """
		"""EACH LINE HAS THE FORM
		TOTAL LINE PERCENT I SHOULD ADD IT MYSELF
		[Fight Date, Gender, Weight Class, Title Fight, Result,Method,Round,Time,Fight Time Format,Fighter 1 Name, Fighter 1 Nickname,
		Fighter 1 Knock Downs, Fighter 1 Sign.Strikes Done,Fighter 1 Sign.Strikes Attempted,
		Fighter 1 Sign.Strikes Perc.,Fighter 1 Total Strikes Done,
		Fighter 1 Total Strikes Attempted,
		Fighter 1 Takedowns Done, Fighter 1 Takedowns Attempted, Fighter 1 Takedowns Perc.,
		Fighter 1 Submission Attempts, Fighter 1 Rev., Fighter 1 Control Time, The same for fighter 2]
		"""

		with open('../../UFC_Fights_Train.tsv','a') as fight_file:
			tsv_writer = csv.writer(fight_file, delimiter='|')
			tsv_writer.writerow(final)
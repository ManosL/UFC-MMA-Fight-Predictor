import string
import scrapy
import re
import csv

event_dates = {}

class FightersSpider(scrapy.Spider):
    name = 'fighters_spider'
    start_urls = ['http://www.ufcstats.com/statistics/events/completed']

    def parse(self,response):
        links = ['http://www.ufcstats.com/statistics/fighters?char=' + l + '&page=all'
                    for l in list(string.ascii_lowercase)]
        
        for link in links:
            yield scrapy.Request(url=link,callback=self.letter_parse)
    
    def letter_parse(self,response):
        
        links = []

        table_rows = response.css('table.b-statistics__table tbody tr.b-statistics__table-row')
        table_rows = table_rows[1:]

        for row in table_rows:
            col = row.css('td.b-statistics__table-col')[0]
            links.append(col.css('a::attr(href)').get())

        print(len(links),response.url)
        for link in links:
            yield scrapy.Request(url = link,callback = self.fighter_parse)

    
    def fighter_parse(self,response):
        final_row = []

        # Getting fighter's name
        pg_title = response.css('h2.b-content__title')

        final_row.append(pg_title.css('span.b-content__title-highlight::text').get())

        final_row[0] = re.sub('\n','',final_row[0])
        final_row[0] = final_row[0].split()
        final_row[0] = ' '.join(list(filter(lambda x: x != '',final_row[0])))

        # Getting fighter's record

        record = pg_title.css('span.b-content__title-record::text').get()
        record = re.sub('Record: ','',record)
        record = re.sub('\n','',record)
        record = re.sub(' ','',record)
        record = re.sub('\(.*?\)','',record)
        record = record.split('-')   # getting a list of form [win,lose,draw]
        record = [int(x) for x in record]

        final_row = final_row + record

        # Details table
        dtls_table = response.css('div.b-fight-details.b-fight-details_margin-top')
        
        # Getting first sub-table
        curr_table = dtls_table.css('div.b-list__info-box.' +
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
        curr_table = dtls_table.css('div.b-list__info-box.b-list__info' + 
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

        #EACH ROW CONTAINS
        #[Fighter Name, Wins, Loses, Draws, Height, Weight, Reach, Stance, DOB, SLpM,
        # Str.Acc., SApM, Str. Def., TD Avg., TD Acc., TD Def., Sub. Avg.]
		
        with open('../../UFC_Fighters.tsv','a') as fight_file:
            tsv_writer = csv.writer(fight_file, delimiter='|')
            tsv_writer.writerow(final_row)
        
        return None
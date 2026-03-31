import re
from scrapy.selector import SelectorList

from UFCStats_Crawlers.spiders.utils import get_fighter_id_from_url

CATCHWEIGHT_CLASS_NAME: str = 'catch weight'

# these weight classes are taken from https://en.wikipedia.org/wiki/Mixed_martial_arts_weight_classes
mma_weight_classes: list[str] = [
    'atomweight', 'strawweight', 'flyweight', 'bantamweight', 'featherweight', 
    'lightweight', 'super lightweight', 'welterweight', 'super welterweight', 
    'middleweight', 'super middleweight', 'light heavyweight', 'cruiserweight', 
    'heavyweight'
]


def process_event_name(event_name: str) -> str:
    # Removing leading and trailing whitespace
    event_name = event_name.strip()

    return event_name


def process_bout_description(bout_desc: str) -> tuple[str, str, str]:
    # Check how you get the weight class because there are other keywords used
    #print(bout_desc)
    bout_desc = re.match(r'<i.*>(?:.*<img.*>)?(.*)</i>', bout_desc, flags=re.DOTALL).groups()[0]
    bout_desc = bout_desc.strip()

    fight_weight_class = CATCHWEIGHT_CLASS_NAME

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
    elif fight_weight_class == CATCHWEIGHT_CLASS_NAME:
        gender = 'unknown'
    else:
        gender = 'male'

    #print("TLF " + str(title_fight) + " WEIGHT CLASS " + weight_class)
    return gender, title_fight, fight_weight_class


def get_fighter_id_name_and_nickname(fighter_html: str) -> tuple[str, str, str]:
    fighter_id = fighter_html.css('''div.b-fight-details__person-text
                                h3.b-fight-details__person-name
                                a::attr(href)''').get().strip()

    if fighter_id is None:
        fighter_id = fighter_html.css('''div.b-fight-details__person-text
                                    h3.b-fight-details__person-name
                                    span.b-link.b-fight-details__person-link::attr(href)''').get().strip()

    fighter_id = get_fighter_id_from_url(fighter_id)

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


def get_fight_result(fighter1_html: str) -> str:
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


def parse_outcome_section(
    outcome_section_selector: SelectorList
) -> tuple[str, str, str, str]:
    outcome = []
    
    outcome.append(
        outcome_section_selector
        .css('i.b-fight-details__text-item_first i::text')
        .getall()[1]
        .strip()
    )

    items = outcome_section_selector \
        .css('i.b-fight-details__text-item') \
        .getall()[0:3]

    for item in items:
        item = re.sub('\n', '', item)
        item = re.sub(r'<i class="b-fight-details__label">.*?</i>', '', item)
        item = re.match(r'<i class="b-fight-details__text-item">(.*)</i>', item).groups()[0]
        item = re.sub(' ', '', item)

        outcome.append(item)

    method, round, time, fight_format = outcome
    return method, round, time, fight_format


def parse_total_stats_table(
    table_selector: SelectorList
) -> tuple[list[str], list[str]]:
    fighter_1_info = []
    fighter_2_info = []

    match_stats_table_columns = table_selector.css('''tr.b-fight-details__table-row
                                            td.b-fight-details__table-col''')[1:]

    for curr_column in match_stats_table_columns:
        rows = curr_column.css('p.b-fight-details__table-text::text').getall()

        stat1 = re.sub(r'\s+', '', rows[0])
        stat2 = re.sub(r'\s+', '', rows[1])

        stat1_groups = re.match(r'(.*)of(.*)', stat1)
        stat2_groups = re.match(r'(.*)of(.*)', stat2)

        if stat1_groups is not None:
            stat1_groups = stat1_groups.groups()
            stat2_groups = stat2_groups.groups()

            fighter_1_info += stat1_groups
            fighter_2_info += stat2_groups
        else:
            fighter_1_info.append(stat1)
            fighter_2_info.append(stat2)
    
    return fighter_1_info, fighter_2_info

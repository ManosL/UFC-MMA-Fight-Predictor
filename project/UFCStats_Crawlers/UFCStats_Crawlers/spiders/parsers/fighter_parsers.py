import re
from typing import Iterator


def parse_fighter_name(name_div: str) -> str:
    fighter_name = re.sub('\n','', name_div)
    fighter_name = fighter_name.split()
    fighter_name = ' '.join(list(filter(lambda x: x != '', fighter_name)))

    return fighter_name


def parse_fighter_record(record_div: str) -> tuple[int, int, int]:
    record = re.sub('Record: ', '', record_div)
    record = re.sub('\n','', record)
    record = re.sub(' ','', record)
    record = re.sub(r'\(.*?\)', '', record)
    record = record.split('-')
    wins, losses, draws = [int(x) for x in record]

    return wins, losses, draws


def __extract_table_element_value(elem: str) -> str:
    extracted = re.sub('\n', '', elem)
    extracted = re.match('<li (.*?)>(.*)</li>', extracted).groups()[1]
    extracted = re.sub('<i (.*)>(.*)</i>', '', extracted)
    extracted = ' '.join(list(filter(lambda x: x != '', extracted.split())))

    return extracted


def parse_measurements_table(table_elems: list[str]) -> Iterator[str]:
    for elem in table_elems:
        extracted = __extract_table_element_value(elem)
        if extracted == '' or extracted == '--':
            extracted = 'No Stat'

        yield extracted


def parse_career_stats_table(table_elems: list[str]) -> Iterator[str]:
    for elem in table_elems:
        extracted = __extract_table_element_value(elem)
        if extracted != '' and extracted != '--':
            yield extracted

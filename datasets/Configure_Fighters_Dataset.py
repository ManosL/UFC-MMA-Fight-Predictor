import pandas as pd
import numpy  as np
from math import isnan

import gender_guesser.detector as gender

def retrieve_initial_dfs(event_df_path, fighters_df_path):
    init_event_df = pd.read_csv(event_df_path, sep = '|', header = None,
    names = ['Fight Date','Gender','Weight Class','Title Fight','Result','Method','Round','Time','Fight Time Format',
        'Fighter 1 ID', 'Fighter 1 Name', 'Fighter 1 Nickname',
		'Fighter 1 Knock Downs', 'Fighter 1 Sign.Strikes Done','Fighter 1 Sign.Strikes Attempted',
		'Fighter 1 Sign.Strikes Perc.','Fighter 1 Total Strikes Done',
		'Fighter 1 Total Strikes Attempted',
		'Fighter 1 Takedowns Done', 'Fighter 1 Takedowns Attempted', 'Fighter 1 Takedowns Perc.',
		'Fighter 1 Submission Attempts', 'Fighter 1 Rev', 'Fighter 1 Control',
        'Fighter 2 ID', 'Fighter 2 Name', 'Fighter 2 Nickname',
		'Fighter 2 Knock Downs', 'Fighter 2 Sign.Strikes Done','Fighter 2 Sign.Strikes Attempted',
		'Fighter 2 Sign.Strikes Perc.','Fighter 2 Total Strikes Done',
		'Fighter 2 Total Strikes Attempted',
		'Fighter 2 Takedowns Done', 'Fighter 2 Takedowns Attempted', 'Fighter 2 Takedowns Perc.',
		'Fighter 2 Submission Attempts', 'Fighter 2 Rev', 'Fighter 2 Control'])

    fighters_df = pd.read_csv(fighters_df_path, sep='|', header=0)

    return init_event_df, fighters_df



def convert_no_stat_to_nan(fighters_df):
    fighters_df = fighters_df.replace('No Stat', np.nan)
    fighters_df = fighters_df.replace('--', np.nan)
    fighters_df = fighters_df.replace('---', np.nan)

    return fighters_df



def convert_percentage_features_to_decimals(fighters_df):
    percentage_features = ['Str.Acc.', 'Str. Def.', 'TD Acc.', 'TD Def.']

    for feature in percentage_features:
        fighters_df[feature] = fighters_df[feature].apply(lambda x: float(x.strip('%')) / 100 if x is not np.nan else np.nan)

    return fighters_df



# Should be an str of form mm:ss
def convert_time_str_to_mins(elem):
    mins = int(elem.split(':')[0])
    secs = int(elem.split(':')[1])

    return_val = (mins * 60 + secs) / 60

    return return_val



def find_average_fight_time(fights_df, fighters_df):
    fighter_ids     = list(fighters_df['Fighter ID'])
    avg_fight_times = []

    for fighter_id in fighter_ids:
        fighters_fights_1 = fights_df[(fights_df['Fighter 1 ID'] == fighter_id)]
        fighters_fights_2 = fights_df[(fights_df['Fighter 2 ID'] == fighter_id)]

        curr_df = pd.concat([fighters_fights_1, fighters_fights_2], axis=0)

        fights_rounds_array = np.array(curr_df['Round'])
        
        last_round_dur       = curr_df['Time']
        last_round_dur_array = np.array(last_round_dur.apply(lambda x: convert_time_str_to_mins(x)))

        fights_duration_array = (fights_rounds_array - 1) * 5 + last_round_dur_array

        if len(fights_duration_array) == 0:
            avg_fight_times.append(0)
        else:
            avg_fight_times.append(fights_duration_array.mean())
    
    fighters_df.insert(5, 'Avg.Time(in Mins)', avg_fight_times)

    return fighters_df 



def find_genders(fights_df, fighters_df):
    # Some special cases from domain knowledge
    full_names_to_gender = {
        'Xie Bin':   'male',
        'Xiao Long': 'male',
        'Fazlo Mulabitinovic': 'male',
        'Kin Moy': 'male',
        'Gisele Moreira': 'female',
        'Aji Susilo': 'male',
        'Kaline Medeiros': 'female',
        'AJ McKee': 'male',
        'Lucasz Sudolski': 'male',
        'AJ Matthews': 'male',
        'Daijiro Matsui': 'male',
        'Waachiim Spiritwolf': 'male',
        'Poppies Martinez': 'male',
        'CJ Marsh': 'male',
        'Kestutis Smirnovas': 'male',
        'Yokthai Sithoar': 'male',
        'AJ Siscoe': 'male',
        'Kaleo Kwan': 'male',
        'Eldari Kurtanidze': 'male',
        'Rizvan Kuniev': 'male',
        'Jorgen Kruth': 'male',
        'Iouri Kotchkine': 'male',
        'Aliev Makhmud': 'male',
        'Klayton Mai': 'male',
        'Maheshate': 'male',
        'Lolohea Mahe': 'male',
        'Abus Magomedov': 'male',
        'Ibragim Magomedov': 'male',
        'Ilima Macfarlane': 'female',
        'Jansey Silva': 'male',
        'Siala Siliga': 'male',
        'Katsuyori Shibata': 'male',
        'Lv Zhenhong': 'male',
        'Sokun Koh': 'male',
        'Jong Man Kim': 'male',
        'Taiei Kin': 'male',
        'Jong Won Kim': 'male',
        'Min Soo Kim': 'male',
        'Jong Wang Kim': 'male',
        'Dae Won Kim': 'male',
        'Aliaskhab Khizriev': 'male',
        'Bu-Kyung Jung': 'male',
        'Jamelle Jones': 'male',
        'Roshaun Jones': 'male',
        'Deshaun Johnson': 'male',
        'Shang Zhifa': 'male',
        'Cleber Luciano': 'male',
        'Rashard Lovelace': 'male',
        'Mabelly Lima': 'female',
        'Mayana Kellem': 'female',
        'Ryo Kawamura': 'male',
        'Canaan Kawaihae': 'male',
        'Bubba Pugh': 'male',
        'Adrienna Jenkins': 'female',
        'Bubba Jenkins': 'male',
        'Gigo Jara': 'male',
        'Matheus Scheffel': 'male',
        'Lumumba Sayers': 'male',
        'Jeimeson Saudino': 'male',
        'Edivan Santos': 'male',
        'Marilia Santos': 'female',
        'Gleristone Santos': 'male',
        'Shaheen Santana': 'male',
        'Jae Suk Lim': 'male',
        'Chi Lewis-Parry': 'male',
        'Lukasz Les': 'male',
        'Cheyden Leialoha': 'male',
        'Ryuta Sakurai': 'male',
        'Thanh Le': 'male',
        'Chatt Lavender': 'male',
        'Yohan Lainesse': 'male',
        'Kemran Lachinov': 'male',
        'Leiticia Pestova': 'female',
        'Jerron Peoples': 'male',
        'Yusup Saadulaev': 'male',
        'Achmed Labasanov': 'male',
        'Jhonoven Pati': 'male',
        'Cyrillo Padilha': 'male',
        'Casey Ryan': 'male',
        'Murilo Rua': 'male',
        'Kain Royer': 'male',
        'Hilarie Rose': 'female',
        'Jonatas Novaes': 'male',
        'Talita Nogueira': 'female',
        'Chidi Njokuani': 'male',
        'Soichi Nishida': 'male',
        'Khomkrit Niimi': 'male',
        'Jaimelene Nievera': 'female',
        'Shungo Oyama': 'male',
        'Kleydson Rodrigues': 'male',
        'Damonte Robinson': 'male',
        'Nasrudin Nasrudinov': 'male',
        'Andrews Nakahara': 'male',
        'Yukiya Naito': 'male',
        'Chibwikem Onyenegecha': 'male',
        'Gadzhi Omargadzhiev': 'male',
        'Casey Olson': 'male',
        'JJ Okanovich': 'male',
        'Michiyoshi Ohara': 'male',
        'Nicdali Rivera-Calanoc': 'female',
        'Vitor Ribeiro': 'male',
        'Iony Razafiarison': 'female',
        'Raou Raou': 'male',
        'JW Wright': 'male',
        'Qiu Lun': 'male',
        'Xue Do Won': 'male',
        'Maimaiti Tuohati': 'male',
        'Teila Tuli': 'male',
        'Blair Tugman': 'male',
        'Dimitiri Wanderley': 'male',
        'Treston Thomison': 'male',
        'Taneisha Tennant': 'female',
        'Ewerton Teixeira': 'male',
        'J.T Taylor': 'male',
        'Junior Tafa': 'male',
        'Milco Voorn': 'male',
        'DeMarco Villalona': 'male',
        'Vitor Vianna': 'male',
        'Jerrel Venetiaan': 'male',
        'Han Ten Yun': 'male',
        'Trenell Young': 'male',
        'Artenus Young': 'male',
        'Dong Sik Yoon': 'male',
        'Sanghoon Yoo': 'male',
        'Goiti Yamauchi': 'male',
        'Ryuki Ueyama': 'male',
        'Zhang Minyang': 'male',
        'Joao Zaiden': 'male',
        'Gleidson Cutis': 'male',
        'Pat Curran': 'male',
        'Luiz Azeredo': 'male',
        'Javy Ayala': 'male',
        'Bazigit Atajev': 'male',
        'Khusein Askhabov': 'male',
        'Chalid Arrab': 'male',
        'JJ Ambrose': 'male',
        'Maiara Amanajas dos Santos': 'female',
        'Estefani Almeida': 'female',
        'Jailton Almeida': 'male',
        'Javi Alanis': 'male',
        'Herdem Alacabek': 'male',
        'Nikk Covert': 'male',
        'JR Coughran': 'male',
        'Jadson Costa': 'male',
        'TJ Cook': 'male',
        'Cortez Coleman': 'male',
        'Coltin Cole': 'male',
        'RJ Clifford': 'male',
        'Mychal Clark': 'male',
        'Hong Man Choi': 'male',
        'Mu Bae Choi': 'male',
        'Gesias Cavalcante': 'male',
        'Bendy Casimir': 'male',
        'Dos Caras Jr.': 'male',
        'Goldman Butler': 'male',
        'Lukasz Brzeski': 'male',
        'Antwain Britt': 'male',
        'Colley Bradford': 'male',
        'Ashe Bowman': 'male',
        'Kyron Bowen': 'male',
        'Francois Botha': 'male',
        'Calen Born': 'male',
        'Kotetsu Boku': 'male',
        'Sherrard Blackledge': 'male',
        'Khadzhi Bestaev': 'male',
        'Pat Benson': 'male',
        'Shonte Barnes': 'male',
        'Junior Barata': 'male',
        'Yohan Banks': 'male',
        'Neiman Gracie': 'male',
        'Ralek Gracie': 'male',
        'Royler Gracie': 'male',
        'Crosley Gracie': 'male',
        'Rickson Gracie': 'male',
        'Tresean Gore': 'male',
        'Tebaris Gordon': 'male',
        'Kier Gooch': 'male',
        'Mikey Gonzalez': 'male',
        'Krishaun Gilmore': 'male',
        'Kultar Gill': 'male',
        'Rulon Gardner': 'male',
        'Fernie Garcia': 'male',
        'Turrell Galloway': 'male',
        'Zelg Galesic': 'male',
        'Alavutdin Gadjiev': 'male',
        'Rilley Dutro': 'male',
        'OJ Dominguez': 'male',
        'AJ Dobson': 'male',
        'Abongo Humphrey': 'male',
        'Casey Huffman': 'male',
        'Blood Diamond': 'male',
        'Jamey-Lyn Horth': 'male',
        'Kailan Hill': 'male',
        'Crezio de Souza': 'male',
        'Johil de Oliveira': 'male',
        'Lemont Davis': 'male',
        'LC Davis': 'male',
        'TJ Hepburn': 'male',
        'Gerric Hayes': 'male',
        'Dhafir Harris': 'male',
        'Baru Harn': 'male',
        'CJ Hamilton': 'male',
        'Yazan Hajeh': 'male',
        'Achilles Estremadura': 'male',
        'Pro Escobedo': 'male',
        'Chel Erwin-Davis': 'male',
        'Sovannahry Em': 'male',
        'Adli Edwards': 'male',
        'Donavon Frelow': 'male',
        'Patricky Freire': 'male',
        'Codale Ford': 'male',
        'Claudionor Fontinelle': 'male',
        'Mal Foki': 'male',
        'AJ Fonseca': 'male',
        'AJ Fletcher': 'male',
        'Isi Fitikefu': 'male',
        'Luiz Firmino': 'male',
        'Erisson Ferreira da Silva': 'male',
        'Bibiano Fernandes': 'male',
        'Rhadi Ferguson': 'male',
        'JP Felty': 'male',
        'Joao Paulo Faria': 'male',
        'Wagnney Fabiano': 'male',
        'Tokimitsu Ishizawa': 'male',
        'Egan Inoue': 'male',
        'Seichi Ikemoto': 'male',
        'Tatsuro Taira': 'male'
    }

    d = gender.Detector()

    first_names = fighters_df['Fighter Name'].apply(lambda x: x.split()[0])
    genders = first_names.apply(lambda x: d.get_gender(x))

    genders = genders.replace('mostly_male', 'male')
    genders = genders.replace('mostly_female', 'female')
    genders = list(genders)

    # For andy or unknown gender fighters we will see what gender they are from their
    # fight, because in there, this information is stored 
    fighters_ids   = list(fighters_df['Fighter ID'])
    fighters_names = list(fighters_df['Fighter Name'])

    for i in range(len(genders)):
        fighter_id     = fighters_ids[i]
        fighter_gender = genders[i]

        if fighter_gender not in ['andy', 'unknown']:
            continue
            
        fighters_fights_1 = fights_df[(fights_df['Fighter 1 ID'] == fighter_id)]
        fighters_fights_2 = fights_df[(fights_df['Fighter 2 ID'] == fighter_id)]

        # If the fighter did not fought previously we cannot determine its gender
        if len(fighters_fights_1) == 0 and len(fighters_fights_2) == 0:
            # If the fighter does not have any fights, our last hope is the dictionary that I hardcoded
            fighter_name = fighters_names[i]

            if fighter_name in full_names_to_gender.keys():
                genders[i] = full_names_to_gender[fighter_name]
        else:
            if len(fighters_fights_1) > 0:
                # A special case for catch weight fights because in this case 
                # the gender is not captured
                if len(fighters_fights_1['Gender'].unique()) == 2:
                    genders[i] = 'female'
                else:
                    genders[i] = list(fighters_fights_1['Gender'])[0]                
            else:
                if len(fighters_fights_2['Gender'].unique()) == 2:
                    genders[i] = 'female'
                else:
                    genders[i] = list(fighters_fights_2['Gender'])[0]        

    genders = pd.Series(genders)
    ambiguous = fighters_df[genders.isin(['andy', 'unknown'])]['Fighter Name']

    if len(ambiguous) > 0:
        print('The following fighters cannot have their gender specified, please complete it yourself, in the generated csv file')
        print('\n'.join(list(ambiguous)))
        print('\nYou should fill their genders in the csv or in the dictionary only in the case you want to use them to predict a fight for them.')

    fighters_df.insert(1, 'Gender', genders)

    return fighters_df



def main():
    fights_datafile_path   = './UFC_Fights_Train.tsv'
    fighters_datafile_path =  './UFC_Fighters.tsv'

    # Do the preprocessing steps
    fights_df, fighters_df = retrieve_initial_dfs(fights_datafile_path, fighters_datafile_path)
    fighters_df = convert_no_stat_to_nan(fighters_df)
    fighters_df = convert_percentage_features_to_decimals(fighters_df)
    fighters_df = find_average_fight_time(fights_df, fighters_df)
    fighters_df = find_genders(fights_df, fighters_df)

    # Write fighters_df to a csv file
    output_path = './Final_Fighters_Dataset.csv'
    
    fighters_df.to_csv(output_path, sep='|', na_rep='NaN', index=False)
    
    return 0

if __name__ == "__main__":
    main()
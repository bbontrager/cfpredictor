from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import os
from pathlib import Path
import tensorflow as tf
import numpy as np



# read JSON config file. put sensitive or configuration settings in there, not hard-coded here

def load_config(config_path='config.json'):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file (default: 'config.json')
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found. "
            f"Please create it based on config.json.example"
        )
    
    with open(config_file, 'r') as f:
        return json.load(f)

# Load configuration
try:
    config = load_config()
    SPREADSHEET_ID = config['spreadsheet_id']
    TRAIN_SHEET= config['training_sheet']
    TRAIN_RANGE= config['training_range']
    CURRENT_SHEET= config['current_sheet']
    CURRENT_RANGE= config['current_range']
    REF_SHEET= config['reference_sheet']
    CONF_RANGE= config['conference_range']
    PLAYOFF_RANGE= config['playoff_rounds_range']
    CFP_RESULT_WEEK= config['cfp_seed_result_week']
    PLAYOFF_RESULT_WEEK= config['playoff_result_week']
    RANK_TOP_N= config['top_n']


except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please create a config.json file with your spreadsheet_id")
    exit(1)
except KeyError:
    print("Error: 'spreadsheet_id' not found in config.json")
    print("Please ensure your config.json contains a 'spreadsheet_id' field")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in config.json: {e}")
    exit(1)

# Version 1 used a range in the spreadsheet to normalize values
# Version 2 uses reads the raw Win/Loss/Rank data and normalizes here


# Define the scope and spreadsheet details
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
TRAIN_RECORDS = f'{TRAIN_SHEET}!{TRAIN_RANGE}'
CURRENT_RECORDS = f'{CURRENT_SHEET}!{CURRENT_RANGE}'

# A = week (number)   (train and results)  1 = preseason, 2=16 = season
# B = Team ID
# C = Team Name
# D = Conference Name  (train, convert to number)
# E = Wins
# F = Losses
# G = AP Rank
# H = Coaches Rank
# I = CFP Rank
# J = eventual CFP Seed  (week 20)
# K = eventual CFP outcome  (week 20)


REF_SHEET='Reference Data'
CONFERENCE_NAMES = f'{REF_SHEET}!{CONF_RANGE}'
PLAYOFF_ROUNDS = f'{REF_SHEET}!{PLAYOFF_RANGE}'



# globals for processing
M1_train_data = None
M2_result_data = None
CONF_ARRAY = None
RESULT_ARRAY = None
CURRENT_ARRAY = None


def get_reference_data():
    """Loading Reference Data"""
    global CONF_ARRAY, RESULT_ARRAY, CURRENT_ARRAY
    conf=read_sheet_data(CONFERENCE_NAMES)
    rounds=read_sheet_data(PLAYOFF_ROUNDS)
    CURRENT_ARRAY=read_sheet_data(CURRENT_RECORDS)

    CONF_ARRAY = [row[1] for row in conf]
    RESULT_ARRAY = [row[1] for row in rounds]

def prepare_season_raw(season):
    """
    Prepare season data WITHOUT normalization - returns raw numeric values.
    Normalization will be handled by TensorFlow layers in the model.
    
    Returns:
        dict: Dictionary with separate arrays for each feature type
    """
    global CFP_RESULT_WEEK, CONF_ARRAY
    
    weeks = []
    conferences = []
    win_pcts = []
    ap_ranks = []
    coaches_ranks = []
    cfp_ranks = []
    
    for row in season:
        # Week (1-16)
        weeks.append(float(row[0]))
        
        # Conference ID (1-based index)
        conferenceID = next((i for i, v in enumerate(CONF_ARRAY) if v == row[3]), None)
        conferences.append(float(conferenceID + 1))
        
        # Win percentage
        if len(row) <= 4 or row[4] == '':
            win_pcts.append(0.0)
        elif float(row[4]) + float(row[5]) == 0:
            win_pcts.append(0.0)
        else:
            win_pcts.append(float(row[4]) / (float(row[4]) + float(row[5])))
        
        # AP Rank (1-25, or 0 for unranked)
        if len(row) <= 6 or row[6] == '':
            ap_ranks.append(0.0)
        else:
            ap_ranks.append(float(row[6]))
        
        # Coaches Rank (1-25, or 0 for unranked)
        if len(row) <= 7 or row[7] == '':
            coaches_ranks.append(0.0)
        else:
            coaches_ranks.append(float(row[7]))
        
        # CFP Rank (1-25, or 0 for unranked)
        if len(row) <= 8 or row[8] == '':
            cfp_ranks.append(0.0)
        else:
            cfp_ranks.append(float(row[8]))
    
    return {
        'weeks': np.array(weeks),
        'conferences': np.array(conferences),
        'win_pcts': np.array(win_pcts),
        'ap_ranks': np.array(ap_ranks),
        'coaches_ranks': np.array(coaches_ranks),
        'cfp_ranks': np.array(cfp_ranks)
    }



def normalize_season(season):
    """
    DEPRECATED: Use prepare_season_raw() instead.
    This function is kept for backward compatibility but should be replaced
    with TensorFlow preprocessing layers in the model.
    
    Process the data (convert to an array of floating points).
    """

    global CFP_RESULT_WEEK, CONF_ARRAY
    float_values = []

    for row in season:
       row_result=[]
       
       # convert week to a float, out of 16 weeks (or CFP_RESULT_WEEK)
       row_result.append(float(row[0])/CFP_RESULT_WEEK)    

       # look up conference ID from a 0-indexed array, and divide by 10 to convert to a 1-indexed float
       conferenceID = next((i for i, v in enumerate(CONF_ARRAY) if v == row[3]), None)
       row_result.append(float(conferenceID+1)/10)    

       # calculate season win/loss percentage, if team has wins or losses
       if len(row)<=4:
          row_result.append(0.)
       elif row[4]=='':
          row_result.append(0.)
       elif float(row[4])+float(row[5])==0:
          row_result.append(0.)
       else:          
          row_result.append(float(row[4])/(float(row[4])+float(row[5])))
       

       # convert AP rank (out of 25) to float, if there is an AP rank, with ranks closer to 1 scoring a higher float
       if len(row)<=6:
          row_result.append(0.)
       elif row[6]=='':
          row_result.append(0.)
       else:          
          row_result.append((float(RANK_TOP_N+1)-float(row[6]))/RANK_TOP_N)
       

       # convert COACHES rank (out of 25) to float, if there is an AP rank, with ranks closer to 1 scoring a higher float
       if len(row)<=7:
          row_result.append(0.)
       elif row[7]=='':
          row_result.append(0.)
       else:          
          row_result.append((float(RANK_TOP_N+1)-float(row[7]))/RANK_TOP_N)
       
       # convert CFP rank (out of 25) to float, if there is an AP rank, with ranks closer to 1 scoring a higher float
       if len(row)<=8:
          row_result.append(0.)
       elif row[8]=='':
          row_result.append(0.)
       else:          
          row_result.append((float(RANK_TOP_N+1)-float(row[8]))/RANK_TOP_N)
       
       float_values.append(row_result)

    return float_values



def create_normalization_layers(training_data):
    """
    Create and adapt normalization layers based on training data.
    Should be called once with training data before building models.
    
    Args:
        training_data: Dict from prepare_season_raw() with raw feature arrays
    
    Returns:
        dict: Dictionary of adapted normalization layers
    """
    normalizers = {}
    
    # Week normalizer (1-16)
    week_norm = tf.keras.layers.Normalization(name='week_norm')
    week_norm.adapt(training_data['weeks'].reshape(-1, 1))
    normalizers['week'] = week_norm
    
    # Conference normalizer
    conf_norm = tf.keras.layers.Normalization(name='conference_norm')
    conf_norm.adapt(training_data['conferences'].reshape(-1, 1))
    normalizers['conference'] = conf_norm
    
    # Win percentage normalizer (already 0-1, but standardize)
    winpct_norm = tf.keras.layers.Normalization(name='winpct_norm')
    winpct_norm.adapt(training_data['win_pcts'].reshape(-1, 1))
    normalizers['win_pct'] = winpct_norm
    
    # Rank normalizers (handle 0 = unranked, 1-25 = ranked)
    # Use inverse ranks so higher rank = higher value
    ap_norm = tf.keras.layers.Normalization(name='ap_rank_norm')
    ap_norm.adapt(training_data['ap_ranks'].reshape(-1, 1))
    normalizers['ap_rank'] = ap_norm
    
    coaches_norm = tf.keras.layers.Normalization(name='coaches_rank_norm')
    coaches_norm.adapt(training_data['coaches_ranks'].reshape(-1, 1))
    normalizers['coaches_rank'] = coaches_norm
    
    cfp_norm = tf.keras.layers.Normalization(name='cfp_rank_norm')
    cfp_norm.adapt(training_data['cfp_ranks'].reshape(-1, 1))
    normalizers['cfp_rank'] = cfp_norm
    
    return normalizers


def normalize_results(results):
    # process the data  (convert to an array of floating points)
    float_values = [[float(col) for col in row] for row in results]

    return float_values

def current_lookup(team_name: str, week: int):
    # look up the current results for a team in the current year for a given week
    target_name = team_name.strip().lower()
    week = int(week)
    
    global CURRENT_ARRAY

    # print(CURRENT_ARRAY)

    resultrow=[]
    result=[]
    for row in CURRENT_ARRAY:
        if row[0] == str(week) and row[2].strip().lower() == target_name:

            resultrow.append(float(row[0]))  # week
            resultrow.append(float(row[1]))  # conf ID
            resultrow.append(float(row[4]) / (float(row[4]) + float(row[5])))  # win pct
            resultrow.append(float(row[5]))  # AP rank
            resultrow.append(float(row[6]))  # coaches tank
            resultrow.append(float(row[7]))  # CFP rank

            result.append(resultrow)
            return result
    
    # Not found
    print(f"No record found for '{team_name.strip()}' in week {week}")
    return None


def get_google_sheets_service():
    """Authenticate and return the Google Sheets API service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    try:
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    except FileNotFoundError:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('sheets', 'v4', credentials=creds)
        return service
    except HttpError as error:
        print(f"An error occurred while creating the service: {error}")
        return None

def read_sheet_data(r):
    """Read data from the specified range in the Google Sheet."""
    service = get_google_sheets_service()
    if not service:
        return None
    
    try:
        # Call the Sheets API to get the values
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=r).execute()
        values = result.get('values', [])
        
        if not values:
            print('No data found in the specified range.')
            return None
        
        return values
       
    
    except HttpError as error:
        print(f"An error occurred while reading the sheet: {error}")
        return None

if __name__ == '__main__':
    get_reference_data()
    #print(CONF_ARRAY)
    #print(RESULT_ARRAY)

    season_train=read_sheet_data(TRAIN_RECORDS)
    #print(season_train)
    

    # New approach: Get raw data for TensorFlow preprocessing
    raw_data = prepare_season_raw(season_train)
    print("Raw data prepared:")
    print(f"  Weeks shape: {raw_data['weeks'].shape}")
    print(f"  AP ranks sample: {raw_data['ap_ranks'][:5]}")

    # Create normalization layers
    normalizers = create_normalization_layers(raw_data)
    print("\nNormalization layers created and adapted to training data")

    # old approach
    normal_train=normalize_season(season_train)
    #print(normal_train)
    # at this point, normal_train is useful for training, equivalent to M1_TRAIN_RANGE_NAME

def init_cfp_model():
    # initialize the model
    get_reference_data()
    season_train=read_sheet_data(TRAIN_RECORDS)
    normal_train=normalize_season(season_train)
    current_season=read_sheet_data(CURRENT_RECORDS)


def get_m1_train_data():
    #data = read_sheet_data(M1_TRAIN_RANGE_NAME)
    #return data
    season_train=read_sheet_data(TRAIN_RECORDS)
    normal_train=normalize_season(season_train)    
    return normal_train

def get_m1_results_rank():
    data = read_sheet_data(M1_TRAIN_RESULTS_RANGE_NAME)
    ranks = []
    for row in data:
        ranks.append(row[0])
    return ranks

def get_m1_results_seed():
    data = read_sheet_data(M1_TRAIN_RESULTS_RANGE_NAME)
    seeds = []
    for row in data:
        seeds.append(row[1])
    return seeds


def get_m1_results_bracket():
    data = read_sheet_data(M1_TRAIN_RESULTS_RANGE_NAME)
    bracket = []
    for row in data:
        bracket.append(row[2])
    return bracket

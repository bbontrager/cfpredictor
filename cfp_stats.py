from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Version 1 used a range in the spreadsheet to normalize values
# Version 2 uses reads the raw Win/Loss/Rank data and normalizes here


# Define the scope and spreadsheet details
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SPREADSHEET_ID = '1twmhwlYZK-k5f-BpY7jM-nPPRmSfqqpeTscSXOszQVw'  # Replace with your Google Sheet ID
TRAIN_SHEET = '2024'
RECORDS_2024 = f'{TRAIN_SHEET}!A2:J860'
CURRENT_SHEET = '2025'
RECORDS_2025 = f'{CURRENT_SHEET}!A2:J860'

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
CONFERENCE_NAMES = f'{REF_SHEET}!F2:G10'
PLAYOFF_ROUNDS = f'{REF_SHEET}!I2:J6'



# globals for processing
M1_train_data = None
M2_result_data = None
CONF_ARRAY = None
RESULT_ARRAY = None
CFP_RESULT_WEEK=16
PLAYOFF_RESULT_WEEK=20


def get_reference_data():
    """Loading Reference Data"""
    global CONF_ARRAY, RESULT_ARRAY
    conf=read_sheet_data(CONFERENCE_NAMES)
    rounds=read_sheet_data(PLAYOFF_ROUNDS)

    CONF_ARRAY = [row[1] for row in conf]
    RESULT_ARRAY = [row[1] for row in rounds]



def normalize_season(season):
    # process the data  (convert to an array of floating points)

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
          row_result.append((26.-float(row[6]))/25)
       

       # convert COACHES rank (out of 25) to float, if there is an AP rank, with ranks closer to 1 scoring a higher float
       if len(row)<=7:
          row_result.append(0.)
       elif row[7]=='':
          row_result.append(0.)
       else:          
          row_result.append((26.-float(row[7]))/25)
       
       # convert CFP rank (out of 25) to float, if there is an AP rank, with ranks closer to 1 scoring a higher float
       if len(row)<=8:
          row_result.append(0.)
       elif row[8]=='':
          row_result.append(0.)
       else:          
          row_result.append((26.-float(row[8]))/25)
       
       float_values.append(row_result)

    return float_values


def normalize_results(results):
    # process the data  (convert to an array of floating points)
    float_values = [[float(col) for col in row] for row in results]

    return float_values

def current_lookup(team_name: str, week: int):
    # look up the current results for a team in the current year for a given week
    target_name = team_name.strip().lower()
    week = int(week)
    
    for row in RECORDS_2025:
        # row[0] = week, row[2] = team_name
        if row[0] == week and row[2].strip().lower() == target_name:
            return row
    
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

    season_24=read_sheet_data(RECORDS_2024)
    #print(season_24)
    normal_24=normalize_season(season_24)
    #print(normal_24)
    # at this point, normal_24 is useful for training, equivalent to M1_TRAIN_RANGE_NAME

def init_cfp_model():
    # initialize the model
    get_reference_data()
    season_24=read_sheet_data(RECORDS_2024)
    normal_24=normalize_season(season_24)
    season_25=read_sheet_data(RECORDS_2025)


def get_m1_train_data():
    #data = read_sheet_data(M1_TRAIN_RANGE_NAME)
    #return data
    season_24=read_sheet_data(RECORDS_2024)
    normal_24=normalize_season(season_24)    
    return normal_24

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

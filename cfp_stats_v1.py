from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Define the scope and spreadsheet details
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SPREADSHEET_ID = '1twmhwlYZK-k5f-BpY7jM-nPPRmSfqqpeTscSXOszQVw'  # Replace with your Google Sheet ID
SHEET_NAME = '2024'
M1_TRAIN_RANGE_NAME = f'{SHEET_NAME}!N2:S860'
M1_TRAIN_RESULTS_RANGE_NAME = f'{SHEET_NAME}!U2:W860'


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
        
        # process the data  (convert to an array of floating points)
        float_values = [[float(col) for col in row] for row in values]

        return float_values
       
    
    except HttpError as error:
        print(f"An error occurred while reading the sheet: {error}")
        return None

if __name__ == '__main__':
    data = read_sheet_data(M1_TRAIN_RANGE_NAME)
    data = read_sheet_data(M1_TRAIN_RESULTS_RANGE_NAME)

def get_m1_train_data():
    data = read_sheet_data(M1_TRAIN_RANGE_NAME)
    return data

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

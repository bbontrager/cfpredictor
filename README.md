# cfpredictor

Machine Learning meets the CFP playoff

This project is a learning sandbox for exploring Machine Learning concepts, tensorflow, and how various approaches to a problem differ in benefits.

It takes data from a Google sheet about NCAA college football records from 2024 and 2025, and uses the 2024 result to predict the outcomes of the 2025 season.

This runs against a specific Google Sheet in a private account.
JSON files for authenticating to Google Sheets are required in the local copy, and must not be committed to the repo.

Configuration settings are saved in config.json.  An example is included in the repo, replace values with your own information. The Google Sheet ID is stored here.


credentials.json
token.json

Model 1 (which is expected to be imprecise due to a small sample size) uses
Week of the season
Conference
Win/Loss Percentage
AP poll rank
Coaches poll rank
CFP rank (in weeks where that poll is released)

Prediction outputs include
Final CFP Rank
Initial Playoff Bracket seeding
CFP Result (National Champion, Runner Up, Made it to semifinals, made it to the second round)



Model 2 (not implemented yet) will also account for
Overall team efficiency (points scored vs points allowed)
Strength of schedule (W/L percent of opponents)
Top 10 Performance (W/L percent against top 10 teams at the time they played)




\# CFP Predictor - Claude Context File



\## Quick Access Links



\### Main Source Files

\- \*\*cfp\_stats.py\*\*: https://raw.githubusercontent.com/bbontrager/cfpredictor/refs/heads/claude/cfp_stats.py

\- \*\*model1.py\*\*: https://raw.githubusercontent.com/bbontrager/cfpredictor/refs/heads/claude/model1.py

\- \*\*pred\_model\_1.py\*\*: https://raw.githubusercontent.com/bbontrager/cfpredictor/refs/heads/claude/pred_model_1.py



\### Configuration

\- \*\*.gitignore\*\*: https://raw.githubusercontent.com/bbontrager/cfpredictor/refs/heads/claude/.gitignore

\- \*\*config.json.example\*\*: https://raw.githubusercontent.com/bbontrager/cfpredictor/refs/heads/claude/config.json.example

\- \*\*README.md\*\*: https://raw.githubusercontent.com/bbontrager/cfpredictor/refs/heads/claude/README.md



\### Repository

\- \*\*Main repo\*\*: https://github.com/bbontrager/cfpredictor

\- \*\*Working branch\*\*: claude

\- \*\*Tree view\*\*: https://github.com/bbontrager/cfpredictor/tree/claude



\## Project Structure



```

cfpredictor/

├── cfp\_stats.py          # Data loading from Google Sheets, normalization

├── model1.py             # TensorFlow model definitions (rank, seed, bracket)

├── pred\_model\_1.py       # Main prediction script

├── config.json           # Local config (gitignored, contains spreadsheet\_id)

├── config.json.example   # Template for configuration

├── credentials.json      # Google API credentials (gitignored)

├── token.json            # Google OAuth token (gitignored)

└── README.md             # Project documentation

```



\## Key Information



\- \*\*Language\*\*: Python

\- \*\*ML Framework\*\*: TensorFlow/Keras

\- \*\*Data Source\*\*: Google Sheets (NCAA football statistics)

\- \*\*Configuration\*\*: JSON file with spreadsheet\_id

\- \*\*Branch for Claude\*\*: `claude`



\## Current Development Focus



\- Configuration management (config.json approach)

\- Model training for CFP predictions

\- Data normalization and preprocessing



\## Notes for Claude



\- All configuration settings should go in config.json (not hardcoded)

\- Sensitive data (credentials, tokens, config.json) are gitignored

\- Use the "claude" branch for all changes


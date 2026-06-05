Movie Recommendation AI Agent
=============================

This is the GitHub/Render-ready version of the project.

Main files:
- app.py
- movies_agent_clean_data_small.csv
- requirements.txt
- templates/index.html
- static/style.css
- data_preparation.py (documents how the full clean dataset was created)

Why the CSV is small:
The original clean dataset was too large for GitHub upload. This version uses the top 6,000 movies by vote_count and popularity, which matches the app limit and keeps the repository lightweight.

Local run:
1. pip install -r requirements.txt
2. Set Gemini API key:
   PowerShell: $env:GEMINI_API_KEY="your_real_key_here"
3. python app.py
4. Open http://127.0.0.1:5000

Render deployment:
1. Upload this folder to GitHub.
2. Create a new Render Web Service from the GitHub repository.
3. Build Command:
   pip install -r requirements.txt
4. Start Command:
   gunicorn app:app
5. Add Environment Variable:
   GEMINI_API_KEY = your_real_key_here
6. Optional Environment Variable:
   GEMINI_MODEL = gemini-2.5-flash

Health check:
Open /health after deployment. If Gemini is configured correctly, you should see:
"gemini_configured": true

Important:
Do not upload your real Gemini API key to GitHub.

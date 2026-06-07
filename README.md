# Movie Recommendation Agent

Files needed in the repository root:
- app.py
- requirements.txt
- movies_agent_clean_data_small.csv
- templates/index.html
- static/style.css

## Render configuration

Build Command:
```bash
pip install -r requirements.txt
```

Start Command:
```bash
gunicorn app:app
```

Environment Variables:
```text
GEMINI_API_KEY=your_real_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-2.0-flash
```

The app tries Gemini models in this order. If one model reaches quota or is unavailable, it automatically tries the next model.

Important: this does not bypass project-wide quota. It only helps when quota is available on another model.

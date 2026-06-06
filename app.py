# -*- coding: utf-8 -*-
"""
Movie Recommendation AI Agent with Gemini API
---------------------------------------------
This Flask app implements a natural-language AI agent for a movie recommendation
student project. The agent:
1. Loads and queries a movie dataset created from MovieLens + TMDB sources.
2. Uses NLP similarity (TF-IDF + cosine similarity) for recommendations.
3. Uses KMeans clustering for movie profile groups.
4. Uses Isolation Forest for anomaly detection.
5. Uses Gemini API to create natural, conversational answers.
6. Answers from the local dataset first. If the dataset does not contain the
   requested movie information, Gemini may answer from general movie knowledge.
7. Refuses questions that are not related to the movie recommendation domain.
8. Handles common typos and approximate movie-title matching.

Run:
    pip install -r requirements.txt
    $env:GEMINI_API_KEY="your_real_key_here"   # PowerShell
    python app.py
"""

import os
import sys
import locale
import ast
import time
import re
import difflib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Force UTF-8 in Windows/PowerShell as much as possible, so Hebrew prompts and
# Gemini responses will not crash because of ASCII encoding.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from google import genai
except Exception:
    genai = None

app = Flask(__name__)
# Keep Hebrew readable in JSON responses returned to the browser.
app.config["JSON_AS_ASCII"] = False
try:
    app.json.ensure_ascii = False
except Exception:
    pass

DATA_PATH = os.getenv("MOVIES_DATA_PATH", "movies_agent_clean_data_small.csv")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_MOVIES_FOR_APP = int(os.getenv("MAX_MOVIES_FOR_APP", "6000"))
NUMERIC_FEATURES = ["runtime", "popularity", "vote_average", "vote_count", "budget", "revenue"]

# ----------------------------------------------------------------------
# 1. Load and clean dataset
# ----------------------------------------------------------------------


def extract_names_from_json(value: object, job_filter: Optional[str] = None) -> List[str]:
    """Extract people names from TMDB-style JSON columns such as cast/crew."""
    if value is None or pd.isna(value):
        return []
    try:
        items = ast.literal_eval(str(value))
        names = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if job_filter and str(item.get("job", "")).lower() != job_filter.lower():
                continue
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def load_movies(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Put movies_agent_clean_data.csv in the project folder "
            "or run data_preparation.py first."
        )

    data = pd.read_csv(path)

    required_text_cols = ["combined_text", "genres_movielens", "title_movielens"]
    for col in required_text_cols:
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)

    if "overview" not in data.columns:
        data["overview"] = ""
    data["overview"] = data["overview"].fillna("").astype(str)

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = 0
        data[col] = pd.to_numeric(data[col], errors="coerce")
        median_value = data[col].median()
        if pd.isna(median_value):
            median_value = 0
        data[col] = data[col].fillna(median_value)

    data = data.drop_duplicates(subset=["title_movielens"]).copy()
    data = data[data["title_movielens"].str.strip() != ""]
    data = data.sort_values(by=["vote_count", "popularity"], ascending=False)
    data = data.head(MAX_MOVIES_FOR_APP).reset_index(drop=True)

    # Lowercase title columns for fast/fuzzy search.
    data["title_lower"] = data["title_movielens"].str.lower()
    data["title_no_year"] = data["title_lower"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()

    # Cast/director handling:
    # The GitHub/Render version uses a smaller CSV that already contains
    # cast_names and director_names instead of the heavy original JSON columns
    # cast and crew. Therefore, preserve those columns if they already exist,
    # and only parse cast/crew JSON as a fallback.
    if "cast_names" in data.columns:
        data["cast_names"] = data["cast_names"].fillna("").astype(str)
    elif "cast" in data.columns:
        data["cast_names"] = data["cast"].apply(lambda value: ", ".join(extract_names_from_json(value)))
    else:
        data["cast_names"] = ""

    if "director_names" in data.columns:
        data["director_names"] = data["director_names"].fillna("").astype(str)
    elif "crew" in data.columns:
        data["director_names"] = data["crew"].apply(lambda value: ", ".join(extract_names_from_json(value, job_filter="Director")))
    else:
        data["director_names"] = ""

    data["cast_names_lower"] = data["cast_names"].str.lower()
    data["director_names_lower"] = data["director_names"].str.lower()

    return data


df = load_movies(DATA_PATH)
all_titles = df["title_movielens"].astype(str).tolist()
all_titles_lower = df["title_lower"].astype(str).tolist()
all_titles_no_year = df["title_no_year"].astype(str).tolist()


# Actor lookup for free-text actor preferences.
all_actor_lookup: Dict[str, str] = {}
for names in df.get("cast_names", pd.Series(dtype=str)).fillna("").astype(str):
    for name in [part.strip() for part in names.split(",") if part.strip()]:
        all_actor_lookup[name.lower()] = name
all_actor_names_lower = list(all_actor_lookup.keys())

ACTOR_ALIASES = {
    "בראד פיט": "Brad Pitt",
    "בראד פיטט": "Brad Pitt",
    "brad pit": "Brad Pitt",
    "brad pitt": "Brad Pitt",
    "ג'ניפר לורנס": "Jennifer Lawrence",
    "ג׳ניפר לורנס": "Jennifer Lawrence",
    "גניפר לורנס": "Jennifer Lawrence",
    "ג'ניפר לורנסס": "Jennifer Lawrence",
    "jennifer lawrence": "Jennifer Lawrence",
    "jenifer lawrence": "Jennifer Lawrence",
    "טום הנקס": "Tom Hanks",
    "tom hanks": "Tom Hanks",
    "ליאונרדו דיקפריו": "Leonardo DiCaprio",
    "לאונרדו דיקפריו": "Leonardo DiCaprio",
    "leonardo dicaprio": "Leonardo DiCaprio",
    "ג'וני דפ": "Johnny Depp",
    "ג׳וני דפ": "Johnny Depp",
    "johnny depp": "Johnny Depp",
}


def extract_requested_actor(user_text: str) -> Optional[str]:
    """Detect an actor/actress preference from Hebrew/English free text.

    The function intentionally returns a name even when the local dataset does
    not contain that actor. That lets the agent move the request to Gemini
    instead of ignoring the actor constraint.
    """
    text = normalize_user_text(user_text).lower() if 'normalize_user_text' in globals() else str(user_text).lower()

    for alias, actor in ACTOR_ALIASES.items():
        if alias.lower() in text:
            return actor

    english_match = re.search(
        r"(?:actor|actress|starring|with)\s+([a-zA-Z][a-zA-Z'\-]+(?:\s+[a-zA-Z][a-zA-Z'\-]+){0,3})",
        text,
    )
    if english_match:
        candidate = english_match.group(1).strip().lower()
        close = difflib.get_close_matches(candidate, all_actor_names_lower, n=1, cutoff=0.78)
        if close:
            return all_actor_lookup.get(close[0])
        return candidate.title()

    # Hebrew patterns such as: "עם גניפר לורנס", "שחקנית ג'ניפר לורנס".
    hebrew_match = re.search(
        r"(?:עם|שחקן|שחקנית|בכיכוב|בכיכובה|בכיכובו)\s+([א-ת׳'\-]+(?:\s+[א-ת׳'\-]+){0,3})",
        text,
    )
    if hebrew_match:
        candidate = hebrew_match.group(1).strip(" .,!?:;\"'")
        # Remove words that are clearly not part of the actor name.
        candidate = re.split(r"\s+(?:באורך|בזאנר|בז׳אנר|עם|ודירוג|דירוג|מעל|מתחת|עד|של)", candidate)[0].strip()
        for alias, actor in ACTOR_ALIASES.items():
            if candidate and (candidate in alias.lower() or alias.lower() in candidate):
                return actor
        return candidate

    return None

# Simple in-memory conversation state per browser/IP.
# This lets the agent ask a clarification question and then use the user's next
# message as a follow-up instead of treating it as a brand-new question.
CONVERSATION_STATE: Dict[str, Dict[str, str]] = {}

# ----------------------------------------------------------------------
# 2. NLP, similarity, clustering and anomaly models
# ----------------------------------------------------------------------

vectorizer = TfidfVectorizer(stop_words="english", max_features=25000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(df[NUMERIC_FEATURES])

svd_components = min(50, max(2, tfidf_matrix.shape[1] - 1))
svd = TruncatedSVD(n_components=svd_components, random_state=42)
text_reduced = svd.fit_transform(tfidf_matrix)
cluster_features = np.hstack([text_reduced, numeric_matrix])

n_clusters = min(8, max(2, len(df) // 400))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(cluster_features)

anomaly_model = IsolationForest(contamination=0.03, random_state=42)
df["anomaly_score"] = anomaly_model.fit_predict(numeric_matrix)
df["is_anomaly"] = df["anomaly_score"] == -1

# ----------------------------------------------------------------------
# 3. Natural language helpers, typo handling and domain guard
# ----------------------------------------------------------------------

GENRE_MAP: Dict[str, List[str]] = {
    "Action": ["action", "akshen", "אקשן"],
    "Adventure": ["adventure", "adventur", "journey", "quest", "הרפתקה", "הרפתקני"],
    "Animation": ["animation", "animated", "cartoon", "אנימציה", "מצויר"],
    "Children": ["children", "kids", "kid", "family", "ילדים", "משפחה"],
    "Comedy": ["comedy", "comdy", "funny", "humor", "laugh", "מצחיק", "קומדיה"],
    "Crime": ["crime", "פשע"],
    "Documentary": ["documentary", "דוקומנטרי"],
    "Drama": ["drama", "emotional", "דרמה", "מרגש"],
    "Fantasy": ["fantasy", "magic", "פנטזיה", "קסם"],
    "Horror": ["horror", "scary", "ghost", "אימה", "מפחיד"],
    "Mystery": ["mystery", "מסתורין"],
    "Romance": ["romance", "romantic", "love", "רומנטי", "אהבה"],
    "Sci-Fi": ["sci-fi", "scifi", "science fiction", "space", "future", "מדע בדיוני", "חלל"],
    "Thriller": ["thriller", "suspense", "מתח"],
    "War": ["war", "battle", "מלחמה"],
    "Musical": ["musical", "music", "מוזיקלי"],
    "IMAX": ["imax"],
}

MOVIE_DOMAIN_KEYWORDS = {
    "movie", "movies", "film", "films", "cinema", "actor", "actress", "director",
    "genre", "plot", "scene", "runtime", "rating", "review", "recommend", "similar",
    "trailer", "watch", "oscars", "imdb", "tmdb", "cluster", "clustering", "trend",
    "trends", "anomaly", "outlier", "popularity", "votes", "cast", "סרט", "סרטים",
    "שחקן", "שחקנית", "במאי", "זאנר", "ז׳אנר", "המלצה", "תמליץ", "דירוג",
    "ביקורת", "ביקורות", "אורך", "דקות", "דומה", "כמו", "מגמות", "חריגות",
    "קלאסטר", "אשכול", "פופולרי", "עלילה"
}

COMMON_TYPOS = {
    "reccomend": "recommend",
    "recomend": "recommend",
    "recommed": "recommend",
    "recomendation": "recommendation",
    "recomendations": "recommendations",
    "similer": "similar",
    "simillar": "similar",
    "smilar": "similar",
    "moovie": "movie",
    "movy": "movie",
    "movi": "movie",
    "flim": "film",
    "genere": "genre",
    "gernre": "genre",
    "runtim": "runtime",
    "lenght": "length",
    "raitng": "rating",
    "revies": "reviews",
}

# Hebrew/transliterated movie title aliases.
# The dataset titles are in English, so this helps requests like:
# "סרט דומה לספיידרמן" -> "Spider-Man".
MOVIE_TITLE_ALIASES = {
    "ספיידרמן": "Spider-Man",
    "ספיידר מן": "Spider-Man",
    "ספיידר מאן": "Spider-Man",
    "ספיידרמן 2": "Spider-Man 2",
    "ספיידר מן 2": "Spider-Man 2",
    "באטמן": "Batman",
    "האביר האפל": "Dark Knight",
    "טיטאניק": "Titanic",
    "אווטאר": "Avatar",
    "צעצוע של סיפור": "Toy Story",
    "צעצועים של סיפור": "Toy Story",
    "הארי פוטר": "Harry Potter",
    "שרק": "Shrek",
    "מטריקס": "Matrix",
    "הנוקמים": "Avengers",
    "מלך האריות": "Lion King",
    "לשבור את הקרח": "Frozen",
    "מהיר ועצבני": "Fast and Furious",
    "פארק היורה": "Jurassic Park",
    "מלחמת הכוכבים": "Star Wars",
    "שודדי הקאריביים": "Pirates of the Caribbean",
    "בחזרה לעתיד": "Back to the Future",
    "פורסט גאמפ": "Forrest Gump",
}


def replace_hebrew_movie_aliases(text: str) -> str:
    updated = str(text)
    for hebrew_title, english_title in MOVIE_TITLE_ALIASES.items():
        updated = re.sub(re.escape(hebrew_title), english_title, updated, flags=re.IGNORECASE)
    return updated


def normalize_user_text(text: str) -> str:
    cleaned = replace_hebrew_movie_aliases(str(text))
    for wrong, correct in COMMON_TYPOS.items():
        cleaned = re.sub(rf"\b{re.escape(wrong)}\b", correct, cleaned, flags=re.IGNORECASE)
    return cleaned


def contains_movie_title(text: str) -> bool:
    text_lower = text.lower()
    words = [w for w in re.findall(r"[a-zA-Z0-9']+", text_lower) if len(w) > 2]
    if not words:
        return False

    # Direct substring check for known titles without year.
    for title in all_titles_no_year[:3000]:
        if len(title) >= 4 and title in text_lower:
            return True

    # Fuzzy check for short title phrases, useful for typos like "Toy Stroy".
    candidate = " ".join(words[-5:])
    close = difflib.get_close_matches(candidate, all_titles_no_year, n=1, cutoff=0.82)
    return len(close) > 0


def is_movie_related(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    tokens = set(re.findall(r"[a-zA-Zא-ת0-9+׳'-]+", text))

    if tokens.intersection(MOVIE_DOMAIN_KEYWORDS):
        return True
    if detect_requested_genres(text):
        return True
    if contains_movie_title(text):
        return True
    return False


def unrelated_response(user_text: str) -> str:
    local = (
        "אני מתמחה בפרויקט הזה בהמלצות סרטים וניתוח דאטה של סרטים 🎬\n"
        "השאלה שכתבת לא נראית קשורה לנושא של המטלה, ולכן לא אענה עליה כאן.\n"
        "אפשר לשאול אותי למשל: recommend a comedy movie, similar to Toy Story, "
        "show trends, detect anomalies, או מה הדירוג של Avatar."
    )
    return call_gemini_for_response(
        user_message=user_text,
        local_result=local,
        task_name="unrelated question guard",
        data_status="unrelated",
        allow_general_knowledge=False,
    ) or local


def detect_requested_genres(user_text: str) -> List[str]:
    text = normalize_user_text(user_text).lower()
    detected = []
    for genre, keywords in GENRE_MAP.items():
        for keyword in keywords:
            pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
            if re.search(pattern, text):
                detected.append(genre)
                break
    return detected


def find_movie_title_in_message(message: str) -> Optional[str]:
    msg = normalize_user_text(message.strip())
    patterns = [
        r"similar to\s+(.+)", r"movies like\s+(.+)", r"movie like\s+(.+)",
        r"like\s+(.+)", r"reviews for\s+(.+)", r"review for\s+(.+)",
        r"rating for\s+(.+)", r"runtime for\s+(.+)", r"how long is\s+(.+)",
        r"who directed\s+(.+)", r"what is\s+(.+?)\s+about",
        r"ביקורות על\s+(.+)", r"ביקורת על\s+(.+)", r"דירוג של\s+(.+)",
        r"דומים ל\s+(.+)", r"דומה ל\s+(.+)", r"כמו\s+(.+)", r"בסגנון\s+(.+)", r"דומה ל(.+)", r"דומים ל(.+)", r"כמו(.+)", r"בסגנון(.+)", r"האורך של\s+(.+)",
        r"מי ביים את\s+(.+)", r"על מה הסרט\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("?.!.,")
    return None


def search_movie_index(movie_title: Optional[str]) -> Optional[int]:
    if not movie_title:
        return None

    title_lower = normalize_user_text(movie_title).lower().strip().strip("?.!.,")
    title_no_year = re.sub(r"\s*\(\d{4}\)", "", title_lower).strip()

    exact = df[df["title_lower"] == title_lower].index
    if len(exact) > 0:
        return int(exact[0])

    exact_no_year = df[df["title_no_year"] == title_no_year].index
    if len(exact_no_year) > 0:
        return int(exact_no_year[0])

    partial = df[df["title_no_year"].str.contains(re.escape(title_no_year), na=False)].index
    if len(partial) > 0:
        return int(partial[0])

    # Fuzzy title matching for typos.
    close = difflib.get_close_matches(title_no_year, all_titles_no_year, n=1, cutoff=0.72)
    if close:
        close_title = close[0]
        idx = df[df["title_no_year"] == close_title].index
        if len(idx) > 0:
            return int(idx[0])

    return None


def extract_possible_movie_title_from_free_text(user_text: str) -> Optional[str]:
    explicit = find_movie_title_in_message(user_text)
    if explicit:
        return explicit

    text = normalize_user_text(user_text).lower()
    # Try to locate any title embedded in the sentence.
    for title in all_titles_no_year[:5000]:
        if len(title) >= 4 and title in text:
            return title

    # Fuzzy match against the whole text after removing common domain words.
    cleaned = re.sub(
        r"\b(movie|film|rating|review|reviews|runtime|duration|recommend|similar|like|about|director|plot|the|a|an|for|of|is|what|who|how|long)\b",
        " ",
        text,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) >= 3:
        close = difflib.get_close_matches(cleaned, all_titles_no_year, n=1, cutoff=0.75)
        if close:
            return close[0]
    return None

# ----------------------------------------------------------------------
# 4. Gemini helper
# ----------------------------------------------------------------------

def get_valid_gemini_api_key() -> Optional[str]:
    """Return a valid-looking Gemini key, or None if Gemini should be skipped.

    This prevents crashes such as:
    'ascii' codec can't encode characters...
    which usually happen when the placeholder Hebrew text is used as the API key
    instead of the real key from Google AI Studio.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")

    invalid_values = {
        "",
        "YOUR_KEY_HERE",
        "your_real_key_here",
        "המפתח_האמיתי_שלך",
        "המפתח האמיתי שלך",
    }

    if api_key in invalid_values:
        return None

    # Google API keys must be ASCII. If the user accidentally leaves Hebrew
    # placeholder text, do not send it to the Gemini SDK.
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        return None

    # Most Gemini keys from AI Studio start with AIza. This is not required for
    # every possible auth method, so we do not reject other ASCII strings, but
    # this check helps catch obvious placeholders.
    if " " in api_key or len(api_key) < 20:
        return None

    return api_key


def sanitize_local_result_for_user(local_result: str) -> str:
    """Hide internal fallback/debug messages from the user."""
    if not local_result:
        return ""
    if local_result.strip().lower().startswith("internal fallback"):
        return ""
    return local_result


def is_temporary_gemini_error(error_text: str) -> bool:
    """Detect temporary Gemini/API overload errors that should be retried."""
    error_text = str(error_text).lower()
    temporary_signals = [
        "503",
        "unavailable",
        "high demand",
        "temporarily",
        "deadline",
        "timeout",
        "rate limit",
        "resource_exhausted",
        "service unavailable",
    ]
    return any(signal in error_text for signal in temporary_signals)


def friendly_gemini_unavailable_message(task_name: str = "") -> str:
    """A natural message instead of exposing technical API errors to the user."""
    return (
        "רגע, אני מנסה להשלים את המידע דרך Gemini אבל השירות עמוס כרגע 😊\n"
        "זה בדרך כלל זמני. נסי לשלוח שוב את אותה בקשה בעוד כמה שניות, "
        "או כתבי לי אם תרצי שאנסה בינתיים לדייק לפי ז׳אנר, אורך, דירוג או שחקן אחר."
    )


def call_gemini_for_response(
    user_message: str,
    local_result: str,
    task_name: str,
    data_status: str = "data_found",
    allow_general_knowledge: bool = False,
) -> Optional[str]:
    """Use Gemini to produce a natural final answer.

    Important:
    - If Gemini is overloaded, the function retries automatically.
    - If Gemini still fails, the user receives a friendly message, not a technical traceback.
    - Internal fallback messages are never shown directly to the user.
    """
    api_key = get_valid_gemini_api_key()
    if not api_key or genai is None:
        return None

    safe_local_result = sanitize_local_result_for_user(local_result)

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are a friendly Movie Recommendation AI Agent for a student ML/AI workshop project.
The project topic is movie recommendations and movie-data analysis.

Important rules:
1. Answer mainly in Hebrew, naturally and conversationally, not like a robotic template.
2. Act like an active AI agent: understand the request, break it into constraints, use the available local result/Gemini knowledge, and guide the user to the next useful step.
3. Understand spelling mistakes and infer the user's intent when reasonable.
4. If data_status is "data_found", base the answer on the local dataset result. You may phrase it naturally and summarize why the results fit, but do not invent extra dataset facts.
5. If data_status is "data_missing" and allow_general_knowledge is true, use your general movie knowledge automatically. Do NOT tell the user that the local dataset was missing information. Do NOT expose internal fallback logic. Just answer naturally as the agent.
6. If the user asked for a recommendation with filters such as actor, genre, runtime, rating, or style, respect ALL of the filters. If an exact match is unrealistic, suggest the closest 2-4 options and explain which constraint is approximate.
7. If data_status is "unrelated", politely say the question is not related to the movie recommendation project and invite the user to ask about movies.
8. Do not answer unrelated topics such as recipes, weather, travel, homework, health, or general life advice.
9. After recommending or identifying a movie, proactively offer one next step, for example: "רוצה שאספר לך תקציר קצר על אחד מהם?" or "רוצה שאמצא משהו דומה אבל קצר יותר?".
10. Keep the answer concise but helpful. Keep movie names in English.
11. Never show internal messages, API errors, stack traces, or implementation details to the user.

User message:
{user_message}

Detected task:
{task_name}

Data status:
{data_status}

Allow general movie knowledge:
{allow_general_knowledge}

Local dataset / local algorithm result:
{safe_local_result}
""".strip()

        last_error = None

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if text and text.strip():
                    return text.strip()

                last_error = "empty Gemini response"

            except UnicodeEncodeError:
                return (
                    "אני לא מצליחה להפעיל את Gemini בגלל בעיית קידוד במפתח או בטקסט שנשלח. "
                    "ודאי שה־GEMINI_API_KEY הוא המפתח האמיתי באנגלית בלבד, בלי טקסט בעברית."
                )

            except Exception as exc:
                last_error = str(exc)

                if is_temporary_gemini_error(last_error):
                    # Short exponential backoff: 1s, 2s, 4s.
                    time.sleep(2 ** attempt)
                    continue

                # Non-temporary errors should not be exposed to the user either.
                return (
                    "אני לא מצליחה להשלים כרגע מידע דרך Gemini. "
                    "נסי שוב בעוד רגע, או נסחי את הבקשה קצת אחרת לפי ז׳אנר, שחקן, אורך או דירוג 😊"
                )

        return friendly_gemini_unavailable_message(task_name)

    except Exception:
        return (
            "אני לא מצליחה להתחבר כרגע ל-Gemini. "
            "נסי שוב בעוד רגע, או בקשי ממני המלצה לפי הדאטה המקומי 🎬"
        )


def final_answer(
    user_message: str,
    local_result: str,
    task_name: str,
    data_status: str = "data_found",
    allow_general_knowledge: bool = False,
) -> str:
    gemini_answer = call_gemini_for_response(
        user_message=user_message,
        local_result=local_result,
        task_name=task_name,
        data_status=data_status,
        allow_general_knowledge=allow_general_knowledge,
    )
    if gemini_answer:
        # If Gemini is temporarily unavailable but the local algorithm already found data,
        # do not hide the useful local recommendation behind an API error message.
        if data_status == "data_found" and any(signal in gemini_answer for signal in ["עמוס", "לא מצליחה להשלים", "לא מצליחה להתחבר"]):
            local_fallback = sanitize_local_result_for_user(local_result)
            if local_fallback:
                return local_fallback
        return gemini_answer

    if data_status == "data_missing" and allow_general_knowledge:
        return (
            "אני צריכה להשלים כאן מידע דרך Gemini, אבל כרגע החיבור ל-Gemini לא פעיל. "
            "ודאי שהגדרת GEMINI_API_KEY אמיתי, ואז הריצי שוב את app.py."
        )

    if data_status == "data_found":
        return sanitize_local_result_for_user(local_result) or (
            "מצאתי תשובה בדאטה, אבל היא לא מוצגת בצורה תקינה. נסי לנסח שוב את הבקשה."
        )

    return sanitize_local_result_for_user(local_result) or (
        "לא הצלחתי לענות על זה כרגע. נסי לנסח שוב את הבקשה סביב סרטים, ז׳אנר, שחקן, אורך או דירוג."
    )

# ----------------------------------------------------------------------
# 5. Formatting functions
# ----------------------------------------------------------------------

def format_movie(row: pd.Series, score: Optional[float] = None, extra: Optional[str] = None) -> str:
    title = row.get("title_movielens", "Unknown Title")
    genres = row.get("genres_movielens", "Unknown")
    rating = float(row.get("vote_average", 0))
    votes = int(float(row.get("vote_count", 0)))
    popularity = float(row.get("popularity", 0))
    runtime = float(row.get("runtime", 0))
    cluster = int(row.get("cluster", -1))

    text = (
        f"🎬 {title}\n"
        f"   Genres: {genres}\n"
        f"   Rating: {rating:.2f} | Votes: {votes} | Popularity: {popularity:.2f}\n"
        f"   Runtime: {runtime:.0f} minutes | Cluster/Profile: {cluster}"
    )
    if score is not None:
        text += f"\n   Match score: {float(score):.3f}"
    if extra:
        text += f"\n   {extra}"
    return text

# ----------------------------------------------------------------------
# 6. Recommendation and analysis functions
# ----------------------------------------------------------------------

def recommend_by_movie(movie_title: Optional[str], n: int = 5) -> Tuple[str, str]:
    movie_index = search_movie_index(movie_title)
    if movie_index is None:
        return (
            "לא מצאתי את הסרט הזה בדאטה המקומי. אם זה סרט אמיתי, Gemini יכול לנסות לעזור מהידע הכללי שלו.",
            "data_missing",
        )

    text_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    numeric_scores = cosine_similarity(numeric_matrix[movie_index].reshape(1, -1), numeric_matrix).flatten()
    same_cluster_bonus = (df["cluster"].values == df.loc[movie_index, "cluster"]).astype(float) * 0.08
    final_scores = 0.72 * text_scores + 0.20 * numeric_scores + same_cluster_bonus

    indices = final_scores.argsort()[::-1]
    results = []
    for idx in indices:
        if idx == movie_index:
            continue
        results.append(format_movie(df.iloc[idx], final_scores[idx]))
        if len(results) == n:
            break

    selected = df.loc[movie_index, "title_movielens"]
    local = (
        f"מצאתי סרטים דומים ל-{selected}.\n"
        "החישוב משלב TF-IDF/Cosine Similarity, מאפיינים מספריים ו-Cluster של הסרט.\n\n"
        + "\n\n".join(results)
    )
    return local, "data_found"


def extract_runtime_filter(text: str) -> Tuple[int, int]:
    text = normalize_user_text(text).lower()
    numbers = [int(x) for x in re.findall(r"\d+", text)]
    min_runtime, max_runtime = 0, 1000

    if "שעה וחצי" in text:
        if any(term in text for term in ["מעל", "יותר"]):
            min_runtime = 90
        elif any(term in text for term in ["מתחת", "פחות", "עד"]):
            max_runtime = 90
        else:
            min_runtime, max_runtime = 80, 110
        return min_runtime, max_runtime
    if "שעה" in text and not numbers:
        if any(term in text for term in ["מעל", "יותר"]):
            min_runtime = 60
        elif any(term in text for term in ["מתחת", "פחות", "עד"]):
            max_runtime = 60
        else:
            min_runtime, max_runtime = 50, 80
        return min_runtime, max_runtime

    if "short" in text or "קצר" in text:
        max_runtime = 90
    elif "long" in text or "ארוך" in text:
        min_runtime = 120
    elif any(term in text for term in ["under", "less than", "עד", "פחות", "מתחת"]):
        if numbers:
            max_runtime = numbers[0]
    elif any(term in text for term in ["over", "more than", "מעל", "יותר"]):
        if numbers:
            min_runtime = numbers[0]
    elif numbers and any(term in text for term in ["אורך", "באורך", "דקות", "minutes", "runtime", "length", "duration"]):
        target = numbers[0]
        min_runtime = max(0, target - 20)
        max_runtime = target + 20
    return min_runtime, max_runtime


def extract_min_rating(text: str) -> Optional[float]:
    text = normalize_user_text(text).lower()
    if any(term in text for term in ["דירוג", "rating", "rated", "ציון"]):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
        if nums:
            valid = [x for x in nums if 0 <= x <= 10]
            if valid:
                return max(valid)
        if any(term in text for term in ["גבוה", "high", "טוב", "מעולה"]):
            return 7.0
        return 6.8
    if any(term in text for term in ["גבוה", "highly rated", "good rating"]):
        return 7.0
    return None


def recommend_by_text(user_text: str, n: int = 5) -> Tuple[str, str]:
    requested_genres = detect_requested_genres(user_text)
    requested_actor = extract_requested_actor(user_text)
    user_vector = vectorizer.transform([normalize_user_text(user_text)])
    text_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    rating_rank = df["vote_average"].rank(pct=True).values
    popularity_rank = df["popularity"].rank(pct=True).values
    vote_rank = df["vote_count"].rank(pct=True).values

    final_scores = 0.70 * text_scores + 0.12 * rating_rank + 0.10 * popularity_rank + 0.08 * vote_rank

    candidate_mask = np.ones(len(df), dtype=bool)

    if requested_genres:
        genre_mask = df["genres_movielens"].apply(
            lambda genres: any(genre in str(genres).split("|") for genre in requested_genres)
        ).values
        final_scores += genre_mask.astype(float) * 0.50
        candidate_mask &= genre_mask

    if requested_actor:
        has_cast_column = "cast_names_lower" in df.columns and df["cast_names_lower"].fillna("").str.strip().ne("").any()
        if not has_cast_column:
            return (f"Internal fallback: cast column missing for {requested_actor}.", "data_missing")

        actor_mask = df["cast_names_lower"].fillna("").str.contains(re.escape(requested_actor.lower()), na=False).values
        if not actor_mask.any():
            return (f"Internal fallback: actor not found locally: {requested_actor}.", "data_missing")
        candidate_mask &= actor_mask
        final_scores += actor_mask.astype(float) * 0.70

    min_runtime, max_runtime = extract_runtime_filter(user_text)
    if min_runtime > 0 or max_runtime < 1000:
        runtime_mask = ((df["runtime"] >= min_runtime) & (df["runtime"] <= max_runtime)).values
        candidate_mask &= runtime_mask

    min_rating = extract_min_rating(user_text)
    if min_rating is not None:
        rating_mask = (df["vote_average"] >= min_rating).values
        candidate_mask &= rating_mask
        final_scores += (df["vote_average"].values / 10.0) * 0.20

    candidates = np.where(candidate_mask)[0]

    if len(candidates) == 0:
        return ("Internal fallback: no full local match for filters.", "data_missing")

    indices = candidates[np.argsort(final_scores[candidates])[::-1]][:n]
    results = [format_movie(df.iloc[idx], final_scores[idx]) for idx in indices]

    details = []
    if requested_genres:
        details.append(f"ז׳אנרים שזוהו: {', '.join(requested_genres)}")
    if requested_actor:
        details.append(f"שחקן/שחקנית: {requested_actor}")
    if min_rating is not None:
        details.append(f"דירוג מינימלי: {min_rating}")
    if min_runtime > 0 or max_runtime < 1000:
        details.append(f"אורך: {min_runtime}-{max_runtime} דקות")

    details_text = "\n" + " | ".join(details) if details else ""
    local = (
        "מעולה, מצאתי התאמות שמתאימות להעדפות שכתבת 😊\n"
        "הנה כמה אפשרויות לפי הדאטה המקומי ומודל הדמיון:"
        + details_text + "\n\n" + "\n\n".join(results)
        + "\n\nרוצה שאבחר לך אחד מהם ואספר עליו תקציר קצר?"
    )
    return local, "data_found"


def recommend_by_runtime(user_text: str, n: int = 5) -> Tuple[str, str]:
    min_runtime, max_runtime = extract_runtime_filter(user_text)
    filtered = df[(df["runtime"] >= min_runtime) & (df["runtime"] <= max_runtime)].copy()
    filtered = filtered.sort_values(by=["vote_average", "popularity", "vote_count"], ascending=False).head(n)
    if filtered.empty:
        return "לא מצאתי סרטים שמתאימים לאורך שביקשת בדאטה המקומי.", "data_missing"

    local = "מצאתי סרטים שמתאימים לפי אורך הסרט שביקשת:\n\n" + "\n\n".join(
        format_movie(row) for _, row in filtered.iterrows()
    )
    return local, "data_found"


def get_movie_info(user_text: str) -> Tuple[str, str]:
    msg = normalize_user_text(user_text).lower()
    title = extract_possible_movie_title_from_free_text(user_text)
    idx = search_movie_index(title)

    if idx is None:
        return ("Internal fallback: movie info not found locally.", "data_missing")

    row = df.iloc[idx]
    movie_title = row.get("title_movielens", "Unknown Title")

    asks_director = "who directed" in msg or "director" in msg or "במאי" in msg or "מי ביים" in msg
    asks_cast = "cast" in msg or "actor" in msg or "actors" in msg or "שחקנים" in msg or "שחקן" in msg or "שחקנית" in msg

    if asks_director:
        for col in ["director_names", "director"]:
            if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                return f"מצאתי בדאטה המקומי: {movie_title} בוים על ידי {row.get(col)}.", "data_found"
        return (f"Internal fallback: director missing for {movie_title}.", "data_missing")

    if asks_cast:
        for col in ["cast_names", "actors"]:
            if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                return f"מצאתי בדאטה המקומי את הקאסט של {movie_title}: {row.get(col)}", "data_found"
        return (f"Internal fallback: cast missing for {movie_title}.", "data_missing")

    overview = row.get("overview", "")
    result = format_movie(row)
    if overview:
        result += f"\n\nOverview from dataset:\n{str(overview)[:700]}"
    return result, "data_found"


def show_trends() -> Tuple[str, str]:
    genres_series = df["genres_movielens"].str.split("|").explode()
    top_genres = genres_series.value_counts().head(10)
    top_movies = df.sort_values(by="popularity", ascending=False).head(5)
    top_clusters = df["cluster"].value_counts().sort_index()

    response = "📊 מגמות ודפוסים בדאטה:\n\n"
    response += "הז׳אנרים הנפוצים ביותר:\n"
    response += "\n".join(f"- {genre}: {count} movies" for genre, count in top_genres.items())
    response += "\n\nהסרטים הפופולריים ביותר:\n"
    response += "\n".join(
        f"- {row['title_movielens']} | Popularity: {float(row['popularity']):.2f} | Rating: {float(row['vote_average']):.2f}"
        for _, row in top_movies.iterrows()
    )
    response += "\n\nהתפלגות Cluster Profiles:\n"
    response += "\n".join(f"- Cluster {cluster}: {count} movies" for cluster, count in top_clusters.items())
    return response, "data_found"


def detect_anomalies() -> Tuple[str, str]:
    anomalies = df[df["is_anomaly"]].copy()
    if anomalies.empty:
        return "לא נמצאו אנומליות משמעותיות לפי Isolation Forest.", "data_found"

    popular_low_rating = anomalies[(anomalies["popularity"] > df["popularity"].quantile(0.85)) & (anomalies["vote_average"] < 5)].head(5)
    high_rating_few_votes = df[(df["vote_average"] >= 8) & (df["vote_count"] < 50)].head(5)
    extreme_runtime = df[df["runtime"] > df["runtime"].quantile(0.97)].sort_values(by="runtime", ascending=False).head(5)

    response = "🔎 זיהוי אנומליות בדאטה:\n\n"
    response += "1. סרטים פופולריים עם דירוג נמוך:\n"
    response += "\n".join(format_movie(row) for _, row in popular_low_rating.iterrows()) if not popular_low_rating.empty else "- לא נמצאו מקרים בולטים."
    response += "\n\n2. דירוג גבוה עם מעט הצבעות:\n"
    response += "\n".join(format_movie(row) for _, row in high_rating_few_votes.iterrows()) if not high_rating_few_votes.empty else "- לא נמצאו מקרים בולטים."
    response += "\n\n3. סרטים ארוכים במיוחד:\n"
    response += "\n".join(format_movie(row) for _, row in extreme_runtime.iterrows()) if not extreme_runtime.empty else "- לא נמצאו מקרים בולטים."
    return response, "data_found"


def describe_clusters() -> Tuple[str, str]:
    response = "🧩 Cluster Analysis - פרופילי סרטים דומים:\n\n"
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        genres = cluster_df["genres_movielens"].str.split("|").explode().value_counts().head(3)
        examples = cluster_df.sort_values(by=["vote_count", "popularity"], ascending=False).head(3)
        response += f"Cluster {cluster_id} | {len(cluster_df)} movies\n"
        response += "Dominant genres: " + ", ".join(genres.index.astype(str)) + "\n"
        response += "Examples: " + ", ".join(examples["title_movielens"].astype(str).tolist()) + "\n\n"
    return response.strip(), "data_found"


def query_gemini_with_web_search(user_message: str) -> str:
    """שולח שאילתה ל-Gemini עם הרשאה ורכיב מובנה לחיפוש חופשי בגוגל (Google Search Grounding)."""
    api_key = get_valid_gemini_api_key()
    if not api_key or genai is None:
        return "מצטער, החיבור ל-Gemini API אינו מוגדר או זמין כרגע."
    try:
        # יוצרים מופע של ה-client עם המפתח התקין
        client_instance = genai.Client(api_key=api_key)
        response = client_instance.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config={
                "tools": [{"google_search": {}}]
            }
        )
        return response.text if response.text else "לא התקבלו תוצאות חיפוש זמינות."
    except Exception as e:
        print(f"Error querying Gemini with search: {e}")
        return "מצטער, חלה שגיאה בניסיון לבדוק את הסרטים המוקרנים כעת בקולנוע."

# ----------------------------------------------------------------------
# 7. Conversational state stubs (כדי למנוע שגיאות אם פונקציות חסרות)
# ----------------------------------------------------------------------
def save_pending_recommendation(session_id, msg):
    if session_id not in CONVERSATION_STATE: CONVERSATION_STATE[session_id] = {}
    CONVERSATION_STATE[session_id]["pending"] = msg

def pop_pending_recommendation(session_id):
    return CONVERSATION_STATE.get(session_id, {}).pop("pending", None)

def save_last_recommendations(session_id, local):
    if session_id not in CONVERSATION_STATE: CONVERSATION_STATE[session_id] = {}
    CONVERSATION_STATE[session_id]["last"] = local

def answer_summary_followup(msg, session_id): return None
def should_ask_followup_before_recommendation(msg): return False
def build_followup_question(msg): return ""

## ----------------------------------------------------------------------
# 8. Agent routing logic (גרסה אוטומטית לחלוטין - מעבר לגוגל כשאין דאטה מקומי)
# ----------------------------------------------------------------------

def movie_agent(user_message: str, session_id: str = "default") -> str:
    original_message = user_message.strip()
    msg = normalize_user_text(original_message).lower().strip()

    if not msg:
        return "כתבי לי איזה סרט בא לך ואנסה להמליץ 😊"

    summary_followup = answer_summary_followup(original_message, session_id)
    if summary_followup:
        return summary_followup

    # --- ברכות ואינטראקציה ראשונית ---
    if msg in ["hi", "hello", "hey", "שלום", "היי", "הי"]:
        local = (
            "היי! אני סוכן AI להמלצות סרטים 🎬\n"
            "אפשר לדבר איתי בשפה חופשית. אני מחפש קודם בדאטה המקומי, "
            "ואם חסר מידע — אני אצא לבדוק עבורך באינטרנט בזמן אמת!\n\n"
            "דוגמאות:\n"
            "- recommend a funny adventure movie\n"
            "- similar to Toy Stroy\n"
            "- איזה סרטים מוקרנים עכשיו בקולנוע?\n"
            "- show trends"
        )
        return final_answer(original_message, local, "greeting", "data_found")

    # תרחיש המשך לשיחה זמנית (אם נשמר קונטקסט)
    pending_original = pop_pending_recommendation(session_id)
    if pending_original:
        combined_request = f"{pending_original}. Follow-up preferences: {original_message}"
        local, status = recommend_by_text(combined_request)
        if status == "data_found":
            save_last_recommendations(session_id, local)
            return final_answer(combined_request, local, "follow-up recommendation", status)
        # אם גם בהמשך השיחה אין מידע מקומי - עוברים אוטומטית לגוגל
        else:
            print("Fallback to Google Search from pending conversation...")
            enhanced_prompt = f"REQUIRED: Use google_search to answer based on the conversation. Context: {combined_request}"
            return query_gemini_with_web_search(enhanced_prompt)

    # הגנת גבולות הדומיין (Guardrails) - מוודא שהשאלה קשורה לעולם הקולנוע
    if not is_movie_related(original_message):
        return unrelated_response(original_message)

    # 1. בדיקת אשכולות (KMeans)
    if any(term in msg for term in ["cluster", "clusters", "clustering", "קלאסטר", "אשכול"]):
        local, status = describe_clusters()
        return final_answer(original_message, local, "clustering", status)

    # 2. בדיקת מגמות וטרנדים
    if any(term in msg for term in ["trend", "trends", "popular", "pattern", "מגמות", "דפוסים", "פופולרי"]):
        local, status = show_trends()
        return final_answer(original_message, local, "trend analysis", status)

    # 3. בדיקת אנומליות (Isolation Forest)
    if any(term in msg for term in ["anomaly", "anomalies", "outlier", "outliers", "חריג", "חריגות", "אנומל"]):
        local, status = detect_anomalies()
        return final_answer(original_message, local, "anomaly detection", status)

    # 4. המלצה לפי סרט דומה (Content-Based)
    if any(term in msg for term in ["similar to", "movies like", "movie like", "דומה ל", "דומים ל", "כמו"]):
        local, status = recommend_by_movie(find_movie_title_in_message(original_message))
        if status == "data_found":
            save_last_recommendations(session_id, local)
            return final_answer(original_message, local, "similarity recommendation", status)
        else:
            # במקום לעבור לידע כללי סגור, עוברים לחיפוש חי בגוגל!
            print("Movie not found locally. Routing to Google Search Grounding...")
            enhanced_prompt = f"REQUIRED: Use google_search to find information and recommendations similar to the movie: {original_message}"
            return query_gemini_with_web_search(enhanced_prompt)

    # 5. סינון לפי אורך סרט (Runtime)
    if any(term in msg for term in ["runtime", "length", "duration", "minutes", "short", "long", "דקות", "קצר", "ארוך", "מתחת", "פחות", "עד", "מעל"]):
        has_filter_word = any(term in msg for term in ["under", "less than", "over", "more than", "short", "long", "מתחת", "פחות", "עד", "מעל", "קצר", "ארוך"])
        if has_filter_word:
            local, status = recommend_by_runtime(original_message)
            if status == "data_found":
                save_last_recommendations(session_id, local)
                return final_answer(original_message, local, "runtime recommendation", status)
        else:
            local, status = get_movie_info(original_message)
            if status == "data_found":
                return final_answer(original_message, local, "movie runtime information", status)
        
        # אם הגענו לכאן וסטטוס המידע חסר
        print("Runtime filter yielded no local results. Routing to Google Search Grounding...")
        enhanced_prompt = f"REQUIRED: Use google_search to look up movies matching this runtime request: {original_message}"
        return query_gemini_with_web_search(enhanced_prompt)

    # 6. מידע ספציפי על סרט (במאי / שחקנים / תקציר)
    if any(term in msg for term in ["review", "reviews", "rating", "ratings", "who directed", "director", "plot", "about", "cast", "summary", "synopsis", "תקציר", "ביקורות", "ביקורת", "דירוג", "במאי", "עלילה"]):
        local, status = get_movie_info(original_message)
        if status == "data_found":
            return final_answer(original_message, local, "movie information / summary", status)
        else:
            print("Specific movie info missing locally. Routing to Google Search Grounding...")
            enhanced_prompt = f"REQUIRED: Use google_search to provide the latest information about: {original_message}"
            return query_gemini_with_web_search(enhanced_prompt)

    # 7. ברירת מחדל: המלצה חופשית (טקסט חופשי / ז'אנר / שחקן)
    local, status = recommend_by_text(original_message)
    if status == "data_found":
        save_last_recommendations(session_id, local)
        return final_answer(original_message, local, "free text recommendation", status)
    
    # 🌟 רשת הביטחון האוטומטית המושלמת: אם האלגוריתם המקומי לא מצא כלום (כולל בקשות על סרטים חדשים בקולנוע)
    print("Local database search resulted in data_missing. Automatically routing to Google Search...")
    enhanced_prompt = (
        f"REQUIRED: You must use your google_search tool to find accurate, real-time answers for this query.\n"
        f"User query: '{original_message}'.\n"
        f"ענה למשתמש בצורה מפורטת, מנומסת וטבעית בעברית על סמך תוצאות החיפוש העדכניות ביותר בגוגל."
    )
    return query_gemini_with_web_search(enhanced_prompt)
# ----------------------------------------------------------------------
# 9. Flask routes
# ----------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")
    session_id = request.headers.get("X-Forwarded-For", request.remote_addr or "local")
    return jsonify({"reply": movie_agent(user_message, session_id=session_id)})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "movies_loaded": len(df),
        "clusters": int(n_clusters),
        "gemini_configured": bool(get_valid_gemini_api_key()) and genai is not None,
    })


if __name__ == "__main__":
    print(f"Loaded {len(df)} movies")
    print(f"Clusters: {n_clusters}")
    print("Gemini configured:", bool(get_valid_gemini_api_key()) and genai is not None)
    if os.getenv("GEMINI_API_KEY") and not get_valid_gemini_api_key():
        print("Gemini API key was ignored because it is missing/invalid.")
    app.run(host="127.0.0.1", port=5000, debug=True)    
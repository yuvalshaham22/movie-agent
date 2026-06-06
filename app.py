# -*- coding: utf-8 -*-
"""
Movie Recommendation AI Agent with Gemini API
---------------------------------------------
This Flask app implements a natural-language AI agent for a movie recommendation
student project. The agent:
1. Loads and queries a movie dataset created from MovieLens + TMDB sources.
2. Uses NLP similarity (TF-IDF + cosine similarity) for recommendations.
3. Uses KMeans clustering for movie profile groups.
4. Uses internal quality signals to support smarter recommendations.
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
import json
import difflib
import json
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

    # Age suitability column. The CSV includes an inferred age_limit column
    # (0/7/13/16/18). If an older CSV is used, create a safe default so the app
    # still runs. age_limit is used for requests such as "מתאים לילדים מתחת לגיל 18".
    if "age_limit" not in data.columns:
        data["age_limit"] = 13
    data["age_limit"] = pd.to_numeric(data["age_limit"], errors="coerce").fillna(13).astype(int)

    if "age_restriction_label" not in data.columns:
        data["age_restriction_label"] = data["age_limit"].apply(lambda x: "לכל המשפחה" if int(x) <= 0 else f"{int(x)}+")
    data["age_restriction_label"] = data["age_restriction_label"].fillna("13+").astype(str)

    if "suitable_under_18" not in data.columns:
        data["suitable_under_18"] = data["age_limit"] < 18

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
# 2. NLP, similarity and clustering models
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
    "trends", "popularity", "votes", "cast", "סרט", "סרטים",
    "שחקן", "שחקנית", "במאי", "זאנר", "ז׳אנר", "המלצה", "תמליץ", "דירוג",
    "ביקורת", "ביקורות", "אורך", "דקות", "דומה", "כמו", "מגמות",
    "קלאסטר", "אשכול", "פופולרי", "עלילה", "גיל", "ילדים", "נוער", "משפחה", "מתאים"
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
    # Hebrew typo backup for cases where Gemini routing is unavailable.
    # Main typo understanding is still handled by Gemini intent routing.
    "עימה": "אימה",
    "איימה": "אימה",
    "אימא": "אימה",
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
        "show trends, סרט דומה לספיידרמן, או מה הדירוג של Avatar."
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
        r"מי משחק(?:ים)?\s+(?:בסרט\s+|ב)?(.+)",
        r"מי משתתף(?:ים)?\s+(?:בסרט\s+|ב)?(.+)",
        r"מי מככב(?:ים)?\s+(?:בסרט\s+|ב)?(.+)",
        r"מי הקאסט של\s+(.+)", r"הקאסט של\s+(.+)", r"השחקנים של\s+(.+)",
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


def build_gemini_constraint_summary(user_message: str) -> str:
    """Create a strict, user-facing constraint summary for Gemini fallback.

    This is intentionally generic: it does not hard-code one actor or one movie.
    It extracts the constraints the local code already knows how to detect and
    passes them to Gemini as mandatory requirements when the local CSV is missing
    a full match.
    """
    constraints = []
    genres = detect_requested_genres(user_message)
    actor = extract_requested_actor(user_message)
    min_runtime, max_runtime = extract_runtime_filter(user_message)
    min_rating = extract_min_rating(user_message)
    min_age, max_age, age_description = extract_age_filter(user_message)

    if genres:
        constraints.append("Requested genre(s): " + ", ".join(genres))
    if actor:
        constraints.append(
            "Requested actor/person: " + actor +
            " — every recommended movie MUST include this actor/person. "
            "If the name is written in Hebrew, infer/translate the well-known English name when possible."
        )
    if min_runtime > 0 or max_runtime < 1000:
        if min_runtime > 0 and max_runtime >= 1000:
            constraints.append(f"Runtime: MUST be longer than or equal to {min_runtime} minutes.")
        elif min_runtime <= 0 and max_runtime < 1000:
            constraints.append(f"Runtime: MUST be shorter than or equal to {max_runtime} minutes.")
        else:
            constraints.append(f"Runtime: MUST be between {min_runtime} and {max_runtime} minutes.")
    if min_rating is not None:
        constraints.append(f"Minimum rating requested: {min_rating}/10 or close to it if exact ratings vary by source.")
    if age_description:
        constraints.append("Age suitability requested: " + age_description)
    if min_age is not None:
        constraints.append(f"Age limit lower bound from local parser: {min_age}+.")
    if max_age is not None:
        constraints.append(f"Age limit upper bound from local parser: up to {max_age}.")

    if not constraints:
        return "No structured constraints detected. Still answer according to the full user message."

    return "\n".join(f"- {item}" for item in constraints)


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
    constraint_summary = build_gemini_constraint_summary(user_message)

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
6. If the user asked for a recommendation with filters such as actor, genre, runtime, rating, age, or style, respect ALL of the filters as mandatory constraints.
7. CRITICAL: Never replace a specific actor/person request with generic genre recommendations. If the user asked for Adam Sandler / אדם סנדלר / any actor, every recommended title must include that actor, unless you explicitly say you could not find a verified match.
8. CRITICAL: Never ignore runtime constraints. If the user asked for מעל 90 דקות / over 90 minutes, do not recommend movies under 90 minutes. If the user asked for under 120 minutes, do not recommend movies over 120 minutes.
9. If an exact match is unrealistic, suggest fewer closest options and clearly say which constraint could not be verified. Do NOT fill the answer with unrelated movies.
10. If data_status is "unrelated", politely say the question is not related to the movie recommendation project and invite the user to ask about movies.
11. Do not answer unrelated topics such as recipes, weather, travel, homework, health, or general life advice.
12. After recommending or identifying a movie, proactively offer one next step, for example: "רוצה שאספר לך תקציר קצר על אחד מהם?" or "רוצה שאמצא משהו דומה אבל קצר יותר?".
13. Keep the answer concise but helpful. Keep movie names in English.
14. Never show internal messages, API errors, stack traces, or implementation details to the user.

User message:
{user_message}

Detected task:
{task_name}

Data status:
{data_status}

Allow general movie knowledge:
{allow_general_knowledge}

Mandatory constraints extracted from the full user message:
{constraint_summary}

Local dataset / local algorithm result:
{safe_local_result}
""".strip()

        try:
            answer = generate_with_gemini_model_fallback(
                client=client,
                contents=prompt,
                use_google_search=False,
            )
            if answer:
                return answer

            return friendly_gemini_unavailable_message(task_name)

        except UnicodeEncodeError:
            return (
                "אני לא מצליחה להפעיל את Gemini בגלל בעיית קידוד במפתח או בטקסט שנשלח. "
                "ודאי שה־GEMINI_API_KEY הוא המפתח האמיתי באנגלית בלבד, בלי טקסט בעברית."
            )

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
    """Return the final user-facing answer.

    Project behavior:
    1. If the local CSV/model found a good answer, return it directly and do not spend a Gemini request.
    2. Only when the local CSV/model cannot answer, use Gemini silently as a fallback.
    3. Never tell the user "I moved from the dataset to Gemini". The transition is internal.
    """
    if data_status == "data_found":
        return sanitize_local_result_for_user(local_result) or (
            "מצאתי תשובה בדאטה, אבל היא לא מוצגת בצורה תקינה. נסי לנסח שוב את הבקשה."
        )

    if data_status == "data_missing" and allow_general_knowledge:
        gemini_answer = call_gemini_for_response(
            user_message=user_message,
            local_result=local_result,
            task_name=task_name,
            data_status=data_status,
            allow_general_knowledge=True,
        )
        if gemini_answer:
            return gemini_answer
        return (
            "לא הצלחתי למצוא תשובה מדויקת כרגע. נסי לנסח את הבקשה קצת אחרת, "
            "למשל לפי ז׳אנר, שחקן, אורך הסרט או סגנון."
        )

    return sanitize_local_result_for_user(local_result) or (
        "לא הצלחתי לענות על זה כרגע. נסי לנסח שוב את הבקשה סביב סרטים, ז׳אנר, שחקן, אורך או דירוג."
    )

# ----------------------------------------------------------------------
# 5. Formatting functions
# ----------------------------------------------------------------------

def format_movie(row: pd.Series, score: Optional[float] = None, extra: Optional[str] = None) -> str:
    """Format a movie for the end user.

    The technical values such as similarity score, cluster id, TF-IDF, cosine
    similarity and internal model details are intentionally not shown here.
    They are part of the project logic, but the user should receive a natural
    recommendation.
    """
    title = row.get("title_movielens", "Unknown Title")
    genres = row.get("genres_movielens", "Unknown")
    rating = float(row.get("vote_average", 0))
    runtime = float(row.get("runtime", 0))
    overview = str(row.get("overview", "")).strip()

    age_label = str(row.get("age_restriction_label", "")).strip()

    text = (
        f"🎬 {title}\n"
        f"ז׳אנרים: {genres}\n"
        f"דירוג: {rating:.1f}/10\n"
        f"אורך הסרט: {runtime:.0f} דקות"
    )

    if age_label:
        text += f"\nמגבלת גיל משוערת: {age_label}"

    if overview:
        short_overview = overview[:220].rsplit(" ", 1)[0]
        if len(overview) > 220:
            short_overview += "..."
        text += f"\nתקציר קצר: {short_overview}"

    if extra:
        text += f"\n{extra}"

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
        f"מצאתי כמה סרטים שיכולים להתאים אם אהבת את {selected}:\n\n"
        + "\n\n".join(results)
        + "\n\nרוצה שאבחר לך מתוכם את הכי מתאים לפי מצב רוח או אורך סרט?"
    )
    return local, "data_found"


def extract_runtime_filter(text: str) -> Tuple[int, int]:
    """Extract a runtime range in minutes from Hebrew/English free text.

    Important cases:
    - "עד 90 דקות" -> max 90
    - "פחות משעתיים" / "עד שעתיים" -> max 120
    - "פחות מ-2 שעות" -> max 120
    - "שעה וחצי" -> around / under / over 90 depending on wording
    """
    text = normalize_user_text(text).lower()
    min_runtime, max_runtime = 0, 1000

    under_terms = ["under", "less than", "עד", "פחות", "מתחת", "לכל היותר", "מקסימום"]
    over_terms = ["over", "more than", "מעל", "יותר", "לפחות", "מינימום"]
    has_under = any(term in text for term in under_terms)
    has_over = any(term in text for term in over_terms)

    # Hebrew hour expressions without relying on raw digits as minutes.
    if "שעה וחצי" in text or "שעה וחצי" in text or "1.5 שעות" in text or "1.5 hour" in text:
        if has_over:
            return 90, 1000
        if has_under:
            return 0, 90
        return 80, 110

    if "שעתיים וחצי" in text or "שעתים וחצי" in text or "2.5 שעות" in text:
        if has_over:
            return 150, 1000
        if has_under:
            return 0, 150
        return 130, 170

    if "שעתיים" in text or "שעתים" in text:
        if has_over:
            return 120, 1000
        if has_under:
            return 0, 120
        return 100, 140

    # Patterns like "2 שעות", "שעתיים", "2 hours".
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:שעות|שעה|hours?|hrs?)", text)
    if hour_match:
        target = int(float(hour_match.group(1)) * 60)
        if has_over:
            return target, 1000
        if has_under:
            return 0, target
        return max(0, target - 20), target + 20

    # Plain "שעה" without a digit.
    if "שעה" in text:
        if has_over:
            return 60, 1000
        if has_under:
            return 0, 60
        return 50, 80

    numbers = [int(x) for x in re.findall(r"\d+", text)]

    if "short" in text or "קצר" in text:
        max_runtime = 90
    elif "long" in text or "ארוך" in text:
        min_runtime = 120
    elif has_under:
        if numbers:
            # If the user wrote "פחות מ-2" near hour words, the hour_match above caught it.
            # Otherwise treat the number as minutes.
            max_runtime = numbers[0]
    elif has_over:
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



def extract_age_filter(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (min_age, max_age, description) based on the user's age suitability request.

    age_limit in the CSV is an inferred minimum recommended age: 0/7/13/16/18.
    For "under 18" we use max_age=17, so 18+ movies are excluded.
    """
    text = normalize_user_text(text).lower()
    min_age = None
    max_age = None
    description = None

    age_context_terms = [
        "גיל", "ילדים", "לילדים", "נוער", "מתבגרים", "משפחה", "משפחתי", "מתאים",
        "age", "kids", "children", "family", "teen", "teens", "under"
    ]
    has_age_context = any(term in text for term in age_context_terms)

    # Explicit under-18 / not adult-only requests.
    if re.search(r"מתחת\s*(?:לגיל|ל)?\s*18", text) or re.search(r"under\s*18", text) or "מתחת ל18" in text:
        max_age = 17
        description = "מתאים מתחת לגיל 18"
    elif re.search(r"עד\s*(?:גיל)?\s*18", text) or "לא 18+" in text or "לא למבוגרים" in text:
        max_age = 17
        description = "ללא סרטי 18+"

    # Kids/family without a specific number: be stricter.
    if max_age is None:
        if any(term in text for term in ["ילדים", "לילדים", "kids", "children", "family", "משפחה", "משפחתי"]):
            max_age = 13
            description = "מתאים לילדים/משפחה"
        elif any(term in text for term in ["נוער", "מתבגרים", "teen", "teens"]):
            max_age = 16
            description = "מתאים לנוער"

    # Numeric age requests, for example "עד גיל 13" or "מעל גיל 18".
    if has_age_context:
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if nums:
            requested_age = nums[0]
            if any(term in text for term in ["מעל", "יותר", "above", "over"]):
                min_age = requested_age
                description = f"מעל גיל {requested_age}"
            elif any(term in text for term in ["מתחת", "under"]):
                max_age = requested_age - 1
                description = f"מתחת לגיל {requested_age}"
            elif any(term in text for term in ["עד", "מקסימום", "max"]):
                max_age = requested_age
                description = f"עד גיל {requested_age}"

    if min_age is None and max_age is None:
        return None, None, None
    return min_age, max_age, description

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

    min_age, max_age, age_description = extract_age_filter(user_text)
    if min_age is not None:
        candidate_mask &= (df["age_limit"] >= min_age).values
    if max_age is not None:
        candidate_mask &= (df["age_limit"] <= max_age).values

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
    if age_description:
        details.append(f"מגבלת גיל: {age_description}")

    details_text = "\n" + " | ".join(details) if details else ""
    local = (
        "מעולה, מצאתי כמה סרטים שיכולים להתאים למה שביקשת 😊"
        + details_text + "\n\n" + "\n\n".join(results)
        + "\n\nרוצה שאבחר לך אחד מהם ואספר עליו תקציר קצר?"
    )
    return local, "data_found"


def recommend_by_runtime(user_text: str, n: int = 5) -> Tuple[str, str]:
    min_runtime, max_runtime = extract_runtime_filter(user_text)
    filtered = df[(df["runtime"] >= min_runtime) & (df["runtime"] <= max_runtime)].copy()
    min_age, max_age, age_description = extract_age_filter(user_text)
    if min_age is not None:
        filtered = filtered[filtered["age_limit"] >= min_age]
    if max_age is not None:
        filtered = filtered[filtered["age_limit"] <= max_age]
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
    asks_cast = (
        "cast" in msg or "actor" in msg or "actors" in msg
        or "שחקנים" in msg or "שחקן" in msg or "שחקנית" in msg
        or "מי משחק" in msg or "משחקים" in msg
        or "מי משתתף" in msg or "משתתפים" in msg
        or "מי מככב" in msg or "מככבים" in msg
        or "קאסט" in msg
    )

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


def model_fallback_list() -> List[str]:
    """Return Gemini models to try in order.

    The first model can be controlled from Render with GEMINI_MODEL.
    Extra fallback models can be controlled from Render with GEMINI_FALLBACK_MODELS,
    for example: gemini-2.5-flash-lite,gemini-2.0-flash

    If one model reaches quota/rate limits or is unavailable, the code automatically
    tries the next model. This does not bypass project-wide quota, but it helps when
    quotas are separate per model.
    """
    configured = os.getenv("GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL

    env_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
    if env_fallbacks:
        models = [configured] + [m.strip() for m in env_fallbacks.split(",") if m.strip()]
    else:
        models = [
            configured,
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
        ]

    unique = []
    for model in models:
        if model and model not in unique:
            unique.append(model)
    return unique


def should_try_next_gemini_model(error_text: str) -> bool:
    """Decide if a Gemini error should trigger trying the next model."""
    text = str(error_text).lower()
    fallback_signals = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "unavailable",
        "503",
        "not_found",
        "404",
        "not supported",
    ]
    return any(signal in text for signal in fallback_signals)


def generate_with_gemini_model_fallback(
    client,
    contents: str,
    use_google_search: bool = False,
):
    """Generate content while trying several Gemini models in order."""
    last_error = None

    for model_name in model_fallback_list():
        try:
            kwargs = {
                "model": model_name,
                "contents": contents,
            }
            if use_google_search:
                kwargs["config"] = {"tools": [{"google_search": {}}]}

            response = client.models.generate_content(**kwargs)
            text = getattr(response, "text", None)
            if text and text.strip():
                print(f"Gemini response generated with model: {model_name}")
                return text.strip()

            last_error = f"empty response from {model_name}"
            print(f"Gemini empty response with model {model_name}")

        except Exception as exc:
            last_error = str(exc)
            print(f"Gemini failed with model {model_name}: {last_error}")
            if should_try_next_gemini_model(last_error):
                continue
            break

    print(f"All Gemini fallback models failed. Last error: {last_error}")
    return None


def query_gemini_with_web_search(user_message: str) -> str:
    """Use Gemini with Google Search grounding only for data that is not in the CSV, such as current cinema movies."""
    api_key = get_valid_gemini_api_key()
    if not api_key or genai is None:
        return (
            "לא הצלחתי לבדוק מידע עדכני כרגע. אפשר לנסות שוב בעוד רגע."
        )

    client_instance = genai.Client(api_key=api_key)

    answer = generate_with_gemini_model_fallback(
        client=client_instance,
        contents=user_message,
        use_google_search=True,
    )
    if answer:
        return clean_cinema_response(answer)

    # If Google Search grounding fails for all models, try regular Gemini fallback.
    # We still avoid falling back to old CSV results for current cinema requests.
    fallback_prompt = (
        "ענה בעברית בצורה טבעית ונעימה לבקשת המשתמש בנושא סרטים בקולנוע. "
        "אם אינך יכול לאמת מידע עדכני או זמני הקרנה, כתוב שהזמינות משתנה בין בתי הקולנוע "
        "ושכדאי לבדוק באתר הקולנוע המקומי. אל תזכיר שגיאות טכניות, מכסות API, מודלים, או פרטי מימוש. "
        "אל תשתמש בכותרת 'למה מתאים'. השתמש בניסוח ידידותי כמו 'תקציר קצר', 'סגנון', ו'הערה'.\n\n"
        + user_message
    )

    answer = generate_with_gemini_model_fallback(
        client=client_instance,
        contents=fallback_prompt,
        use_google_search=False,
    )
    if answer:
        return clean_cinema_response(answer)

    return (
        "לא הצלחתי לבדוק מידע עדכני כרגע. אפשר לנסות שוב בעוד רגע."
    )



# ----------------------------------------------------------------------
# 7. Conversation context + Gemini intent routing helpers
# ----------------------------------------------------------------------

def get_state(session_id: str) -> Dict[str, str]:
    if session_id not in CONVERSATION_STATE:
        CONVERSATION_STATE[session_id] = {}
    return CONVERSATION_STATE[session_id]


def save_last_recommendation_request(session_id: str, request_text: str) -> None:
    state = get_state(session_id)
    state["last_recommendation_request"] = request_text
    state["last_active_movie_context"] = request_text


def get_last_recommendation_request(session_id: str) -> str:
    return get_state(session_id).get("last_recommendation_request", "")


def save_last_home_request(session_id: str, request_text: str) -> None:
    """Save the latest home-viewing request and keep it as active context."""
    state = get_state(session_id)
    state["last_home_request"] = request_text
    state["last_recommendation_request"] = request_text
    state["last_active_movie_context"] = request_text
    state["last_viewing_mode"] = "home"


def get_last_home_request(session_id: str) -> str:
    return get_state(session_id).get("last_home_request", "")


def save_last_active_context(session_id: str, request_text: str) -> None:
    if request_text:
        get_state(session_id)["last_active_movie_context"] = request_text


def get_last_active_context(session_id: str) -> str:
    state = get_state(session_id)
    return (
        state.get("last_active_movie_context")
        or state.get("last_home_request")
        or state.get("last_cinema_request")
        or state.get("last_recommendation_request")
        or ""
    )


def save_viewing_mode_pending(session_id: str, request_text: str) -> None:
    state = get_state(session_id)
    state["pending_viewing_request"] = request_text
    state["last_recommendation_request"] = request_text


def pop_viewing_mode_pending(session_id: str) -> Optional[str]:
    state = get_state(session_id)
    return state.pop("pending_viewing_request", None)


def save_last_cinema_request(session_id: str, request_text: str) -> None:
    state = get_state(session_id)
    state["last_cinema_request"] = request_text
    state["last_active_movie_context"] = request_text
    state["last_viewing_mode"] = "cinema"


def get_last_cinema_request(session_id: str) -> str:
    return get_state(session_id).get("last_cinema_request", "")


def save_last_cinema_answer(session_id: str, answer_text: str) -> None:
    """Save the last user-facing cinema answer so follow-up questions can refer to it.

    Example: after the agent recommends "חוות החיות", the next message may be
    "מי השחקנים בסרט חוות החיות". The router needs the previous answer text,
    not only the original request "אני רוצה סרט אימה".
    """
    if not answer_text:
        return
    state = get_state(session_id)
    # Keep the context compact so Gemini receives the useful titles/details without
    # making the prompt too long.
    state["last_cinema_answer"] = str(answer_text)[:2500]


def get_last_cinema_answer(session_id: str) -> str:
    return get_state(session_id).get("last_cinema_answer", "")


def answer_cinema_request(session_id: str, full_cinema_request: str) -> str:
    """Run the current-cinema flow and save the answer for later follow-ups."""
    save_last_cinema_request(session_id, full_cinema_request)
    answer = query_gemini_with_web_search(build_cinema_search_prompt(full_cinema_request))
    save_last_cinema_answer(session_id, answer)
    save_last_recommendations(session_id, answer)
    return answer


def get_last_viewing_mode(session_id: str) -> str:
    return get_state(session_id).get("last_viewing_mode", "")


def extract_recommended_titles(local_result: str) -> List[str]:
    titles = re.findall(r"🎬\s*(.+?)\n", local_result or "")
    return [title.strip() for title in titles if title.strip()]


def save_last_recommendations(session_id: str, local_result: str) -> None:
    titles = extract_recommended_titles(local_result)
    if titles:
        state = get_state(session_id)
        state["last_recommended_titles"] = "||".join(titles[:5])


def get_last_recommended_titles(session_id: str) -> List[str]:
    raw = get_state(session_id).get("last_recommended_titles", "")
    return [title for title in raw.split("||") if title]


def user_accepts_summary_offer(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower().strip()
    yes_terms = ["כן", "יאללה", "בטח", "סבבה", "אפשר", "ספר", "ספרי", "תקציר", "על הראשון", "yes", "sure", "ok", "summary", "synopsis"]
    return any(term in text for term in yes_terms)


def answer_summary_followup(user_text: str, session_id: str) -> Optional[str]:
    titles = get_last_recommended_titles(session_id)
    if not titles or not user_accepts_summary_offer(user_text):
        return None
    selected_title = titles[0]
    lower_text = user_text.lower()
    for title in titles:
        if re.sub(r"\s*\(\d{4}\)", "", title).lower() in lower_text:
            selected_title = title
            break
    local = (
        f"The user accepted the offer to expand information about {selected_title}. "
        "Answer in Hebrew with user-facing wording only. Include concise basic information if known: "
        "short plot, main actors, approximate age rating, release year, and whether it has sequels/franchise context. "
        "Do not write internal headings like 'why it fits'."
    )
    return final_answer(user_text, local, "proactive movie summary", "data_missing", allow_general_knowledge=True)


def has_recommendation_intent(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    return any(term in text for term in [
        "recommend", "recommendation", "movie", "film", "בא לי", "רוצה", "תמליץ", "המלצה", "סרט", "סרטים"
    ]) or bool(detect_requested_genres(text))

def has_specific_recommendation_filters(user_text: str) -> bool:
    """Return True when the user gave enough details to recommend from CSV.

    A very broad request like "אני אשמח להמלצה על סרט" should not return
    the global top-ranked movies, because that feels arbitrary to the user.
    In that case the agent should ask a short clarification question.
    """
    text = normalize_user_text(user_text).lower()

    if detect_requested_genres(text):
        return True
    if extract_runtime_filter(text) != (0, 1000):
        return True
    if extract_min_rating(text) is not None:
        return True
    if extract_requested_actor(text) is not None:
        return True
    min_age, max_age, _ = extract_age_filter(text)
    if min_age is not None or max_age is not None:
        return True
    if find_movie_title_in_message(user_text) or extract_possible_movie_title_from_free_text(user_text):
        return True
    if any(term in text for term in [
        "דומה", "דומים", "כמו", "בסגנון", "similar", "like",
        "קולנוע", "הקרנה", "מוקרנים", "מוקרן", "עכשיו", "cinema", "theater", "now playing",
        "טרנד", "מגמות", "trend", "popular", "פופולרי"
    ]):
        return True

    return False


def should_ask_for_recommendation_details(user_text: str) -> bool:
    """Ask for preferences when the user only asks for a generic movie recommendation."""
    text = normalize_user_text(user_text).lower().strip()
    if not has_recommendation_intent(text):
        return False
    if has_specific_recommendation_filters(text):
        return False

    generic_phrases = [
        "המלצה על סרט", "תמליץ", "תמליצי", "בא לי סרט", "רוצה סרט",
        "אשמח להמלצה", "recommend a movie", "movie recommendation", "recommend me a movie"
    ]
    return any(phrase in text for phrase in generic_phrases) or text in ["סרט", "סרטים", "movie", "film"]


def build_recommendation_details_question() -> str:
    return (
        "בשמחה 😊 כדי שההמלצה תהיה באמת רלוונטית ולא סתם סרט כללי, "
        "אפשר לדייק לי קצת מה בא לך לראות?\n\n"
        "למשל:\n"
        "• ז׳אנר: קומדיה, מתח, אקשן, אימה, רומנטי\n"
        "• אורך: עד 90 דקות / לא משנה\n"
        "• קהל יעד: מתאים לילדים / מעל גיל 18\n"
        "• צפייה: בבית או בקולנוע\n\n"
        "אפשר לכתוב חופשי, למשל: ‘סרט מצחיק עד 90 דקות’ או ‘סרט מתח בקולנוע’."
    )


def should_ask_home_or_cinema(user_text: str) -> bool:
    """Ask home/cinema for recommendation requests only, not detail follow-ups.

    A question like "מי השחקנים בסרט חוות החיות" contains the word "סרט",
    but it is not a new recommendation request. It should be answered from the
    existing conversation context instead of asking again "בית או קולנוע".

    The user experience we want:
    - User gives a preference, e.g. "סרט עימה באורך שעה וחצי" or "סרט מצחיק עד שעתיים".
    - Agent first asks whether the user wants home viewing or cinema.
    - If home: CSV first, then Gemini only if CSV cannot answer.
    - If cinema: Gemini Search / public cinema information.
    """
    if 'is_movie_detail_followup' in globals() and is_movie_detail_followup(user_text):
        return False

    text = normalize_user_text(user_text).lower()
    if not has_recommendation_intent(text):
        return False
    if any(term in text for term in ["קולנוע", "הקרנה", "הקרנות", "מוקרנים", "מוקרן", "עכשיו", "cinema", "theater", "theatre", "now playing"]):
        return False
    if any(term in text for term in ["בבית", "טלוויזיה", "טלויזיה", "סטרימינג", "home", "streaming", "watch at home"]):
        return False
    if any(term in text for term in ["דומה", "דומים", "כמו", "similar", "movies like", "movie like", "trend", "cluster", "דירוג של", "במאי"]):
        return False
    return True


def build_home_or_cinema_question(user_text: str) -> str:
    genres = detect_requested_genres(user_text)
    min_runtime, max_runtime = extract_runtime_filter(user_text)
    min_age, max_age, age_description = extract_age_filter(user_text)

    details = []
    if genres:
        details.append(f"ז׳אנר: {genres[0]}")
    actor = extract_requested_actor(user_text)
    if actor:
        details.append(f"שחקן/שחקנית: {actor}")
    if min_runtime > 0 or max_runtime < 1000:
        details.append(f"אורך: {min_runtime}-{max_runtime} דקות")
    if age_description:
        details.append(f"מגבלת גיל: {age_description}")

    details_text = "\n" + " | ".join(details) if details else ""
    return (
        "מעולה 😊 כדי לתת לך המלצה רלוונטית, איפה תרצי/תרצה לראות את הסרט?"
        + details_text +
        "\n\nאפשר לבחור:\n"
        "• בבית — אני אחפש קודם במאגר הסרטים שלנו, ואם חסר מידע אשלים .\n"
        "• בקולנוע — אני אבדוק מידע עדכני על סרטים שמוקרנים עכשיו.\n\n"
        "אפשר לענות פשוט: \"בבית\" או \"בקולנוע\"."
    )


def build_context_for_router(session_id: str) -> str:
    state = get_state(session_id)
    parts = []
    if state.get("last_active_movie_context"):
        parts.append("Active movie context: " + state["last_active_movie_context"])
    if state.get("last_recommendation_request"):
        parts.append("Last recommendation request: " + state["last_recommendation_request"])
    if state.get("last_home_request"):
        parts.append("Last home-viewing request: " + state["last_home_request"])
    if state.get("last_cinema_request"):
        parts.append("Last cinema/current request: " + state["last_cinema_request"])
    if state.get("last_cinema_answer"):
        parts.append("Last cinema/current answer shown to user: " + state["last_cinema_answer"])
    if state.get("last_viewing_mode"):
        parts.append("Last viewing mode: " + state["last_viewing_mode"])
    if state.get("last_recommended_titles"):
        parts.append("Last recommended titles: " + state["last_recommended_titles"].replace("||", ", "))
    if state.get("pending_viewing_request"):
        parts.append("Pending home/cinema question for: " + state["pending_viewing_request"])
    return "\n".join(parts)


def is_short_followup_answer(user_text: str) -> bool:
    return len(str(user_text).split()) <= 6


def is_contextual_refinement(user_text: str) -> bool:
    """A follow-up that adds constraints to the previous movie/cinema request."""
    text = normalize_user_text(user_text).lower()
    refinement_terms = [
        "מתחת", "מעל", "גיל", "ילדים", "לילדים", "נוער", "18", "16", "13", "12",
        "מדובב", "עברית", "בלי", "עם", "יותר", "פחות", "קצר", "ארוך", "דירוג", "מתאים",
        "אימה", "מתח", "אקשן", "קומדיה", "רומנטי", "משפחה", "thriller", "horror", "action", "comedy"
    ]
    return any(term in text for term in refinement_terms) or bool(detect_requested_genres(text))


def clean_cinema_response(text: str) -> str:
    """Make Gemini output look user-facing, not like internal reasoning."""
    if not text:
        return text
    replacements = {
        "למה זה מתאים לבקשה": "תקציר קצר",
        "למה זה מתאים": "תקציר קצר",
        "למה מתאים לבקשה": "תקציר קצר",
        "למה מתאים": "תקציר קצר",
        "Why it fits": "תקציר קצר",
        "Why this fits": "תקציר קצר",
        "שימו לב": "הערה",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned.strip()


def safe_json_loads(text: str) -> Dict[str, object]:
    try:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return ast.literal_eval(match.group(0)) if "'" in match.group(0) and '"' not in match.group(0) else __import__("json").loads(match.group(0))
    except Exception:
        pass
    return {}


def get_routing_analysis(user_message: str, conversation_context: str = "") -> Dict[str, object]:
    """Let Gemini understand Hebrew typos and follow-up intent.

    The router only decides where to answer from:
    - CSV for normal historical/movie-dataset recommendations.
    - Gemini Search for current cinema / now playing requests.
    """
    api_key = get_valid_gemini_api_key()
    prompt = f"""
You are an intent router for a Hebrew movie recommendation agent.
Understand Hebrew, English, typos, transliteration, and short follow-up answers.
Return ONLY valid JSON, no markdown.

Conversation context:
{conversation_context or "none"}

Current user message:
{user_message}

Decide:
- is_movie_related: true/false
- viewing_preference: "home", "cinema", or "unclear". Treat Hebrew typos like קוללנוע/קלונוע/קולנוע as cinema if the context is movie viewing.
- needs_current_external_info: true only for current cinema screenings, now playing, showtimes, new releases, or other live information not in a CSV.
- requested_genre: English genre if clear, e.g. Thriller, Horror, Action, Comedy, otherwise "". In a movie context, Hebrew typos such as "עימה" or "איימה" usually mean "אימה" / Horror.
- normalized_query_hebrew: rewrite the FULL user intent in Hebrew using the conversation context. Correct spelling mistakes, especially movie-domain typos, and preserve ALL important constraints such as genre, age, under 18, cinema, home, actor, runtime, and style. Examples: "עימה באורך שעה וחצי" -> "סרט אימה באורך שעה וחצי".
- user_task: one of "recommendation", "movie_details", "similar_movies", "data_analysis", "other_movie", or "unrelated".
- detail_question: true if the user asks for factual information about a specific movie/person/movie topic, such as cast, actors, who plays, director, plot, age rating, release year, sequels, runtime, reviews, or genre.
- recommendation_request: true only if the user is asking you to recommend/select/find movies to watch.
- should_ask_viewing_mode: true only if this is a recommendation_request and the user did not already specify home/cinema/current screenings.
""".strip()

    if api_key and genai is not None:
        try:
            client = genai.Client(api_key=api_key)
            router_text = generate_with_gemini_model_fallback(
                client=client,
                contents=prompt,
                use_google_search=False,
            )
            data = safe_json_loads(router_text or "")
            if data:
                return data
        except Exception as e:
            print(f"Gemini routing failed, using local backup router: {e}")

    # Backup only if Gemini routing is unavailable / quota exhausted.
    # IMPORTANT: In a follow-up after asking "home or cinema", the context contains
    # the word "cinema" because it describes the question we asked. Therefore the
    # user's CURRENT answer must get priority. Otherwise an answer like "בבית"
    # may be misrouted to current cinema search just because the context says
    # "current cinema screenings".
    text = normalize_user_text(user_message).lower()
    context = normalize_user_text(conversation_context).lower()
    combined = f"{context} {text}"

    home_like_current = any(term in text for term in [
        "בבית", "בית", "טלוויזיה", "טלויזיה", "סטרימינג", "לצפייה בבית",
        "home", "streaming", "watch at home"
    ])
    cinema_like_current = any(term in text for term in [
        "קולנוע", "הקרנה", "הקרנות", "מוקרנים", "מוקרן", "עכשיו",
        "cinema", "theater", "theatre", "now playing"
    ])

    # Contextual cinema is used only when the user did not explicitly choose home.
    cinema_like_context = any(term in combined for term in [
        "קולנוע", "הקרנה", "הקרנות", "מוקרנים", "מוקרן", "עכשיו",
        "cinema", "theater", "theatre", "now playing"
    ])

    if home_like_current:
        viewing_preference = "home"
        needs_current_external_info = False
    elif cinema_like_current:
        viewing_preference = "cinema"
        needs_current_external_info = True
    elif cinema_like_context and any(term in text for term in ["כן", "yes", "יאללה", "סבבה", "אפשר"]):
        viewing_preference = "cinema"
        needs_current_external_info = True
    else:
        viewing_preference = "unclear"
        needs_current_external_info = False

    genres = detect_requested_genres(combined)
    detail_question = is_movie_detail_followup(user_message) or bool(extract_possible_movie_title_from_free_text(user_message) and not has_recommendation_intent(user_message))
    recommendation_request = has_recommendation_intent(user_message) and not detail_question
    user_task = "movie_details" if detail_question else ("recommendation" if recommendation_request else ("other_movie" if is_movie_related(combined) else "unrelated"))
    return {
        "is_movie_related": bool(is_movie_related(combined) or cinema_like_current or home_like_current or genres),
        "viewing_preference": viewing_preference,
        "needs_current_external_info": needs_current_external_info,
        "requested_genre": genres[0] if genres else "",
        "normalized_query_hebrew": combined.strip() or user_message,
        "user_task": user_task,
        "detail_question": detail_question,
        "recommendation_request": recommendation_request,
        "should_ask_viewing_mode": bool(recommendation_request and viewing_preference == "unclear" and not needs_current_external_info),
    }



def summarize_detected_constraints_for_prompt(full_request: str) -> str:
    """Create explicit hard constraints for Gemini search prompts.

    This is user-facing logic support: the model still does the live lookup, but
    the prompt receives the original constraints in a clear structured way so it
    does not forget genre/runtime/age from previous turns.
    """
    normalized = normalize_user_text(full_request)
    genres = detect_requested_genres(normalized)
    min_runtime, max_runtime = extract_runtime_filter(normalized)
    min_age, max_age, age_description = extract_age_filter(normalized)

    constraints = []
    if genres:
        constraints.append("Requested genre(s): " + ", ".join(genres))
    if min_runtime > 0 or max_runtime < 1000:
        constraints.append(f"Runtime: {min_runtime}-{max_runtime} minutes")
    if age_description:
        constraints.append("Age suitability: " + age_description)
    if any(term in normalize_user_text(full_request).lower() for term in ["קולנוע", "מוקרן", "מוקרנים", "הקרנה", "now playing", "cinema", "theater", "theatre"]):
        constraints.append("Viewing mode: current cinema / now playing")
    return "\n".join(constraints) if constraints else "No structured constraints detected; infer carefully from the full request."


def is_movie_detail_followup(user_text: str) -> bool:
    """Detect follow-up questions about a specific recommended/current movie."""
    text = normalize_user_text(user_text).lower()
    detail_terms = [
        "מי השחקנים", "מי משחק", "מי משחקים", "משחקים", "מי משתתף", "מי משתתפים",
        "משתתפים", "מי מככב", "מי מככבים", "מככבים", "שחקנים", "קאסט", "cast", "actors", "actor",
        "במאי", "מי ביים", "director", "שנת יציאה", "מתי יצא", "release year",
        "מגבלת גיל", "הגבלת גיל", "גיל", "סרט המשך", "סרטי המשך", "sequel", "sequels",
        "על מה", "תקציר", "summary", "synopsis", "פרטים", "מידע"
    ]
    return any(term in text for term in detail_terms)


def build_movie_detail_search_prompt(user_message: str, conversation_context: str = "") -> str:
    return f"""
You are a Hebrew movie recommendation agent with Google Search grounding.
The user is asking for details about a movie mentioned in the conversation or in current cinema results.
Use Google Search when available, prioritize reliable public sources, and answer only the asked question.

Conversation context:
{conversation_context or "none"}

Current user question:
{user_message}

Rules:
1. Infer the movie title from the current question and conversation context. Hebrew titles may be local translations. Prefer titles that appeared in the last answer shown to the user.
2. If the user asks for actors/cast, answer with the main actors only.
3. If the user asks for age rating, release year, sequels, director, or short plot, provide concise basic information.
4. Do NOT switch to unrelated recommendations.
5. If you cannot verify the exact information, say that it is not fully verified and suggest checking the cinema/movie page.
6. Answer in natural Hebrew, without internal implementation details.
""".strip()



def answer_movie_related_with_gemini(user_message: str, conversation_context: str = "") -> str:
    """General fallback for movie-domain questions that the local CSV cannot answer.

This prevents adding endless hard-coded Hebrew phrases such as "מי משחק",
"מי משתתף", etc. If Gemini/router understood that the question is about
movies, but the CSV has no exact answer, Gemini answers from general movie
knowledge / Google Search grounding.
"""
    prompt = f"""
You are a Hebrew Movie Recommendation AI Agent.
The user asked a question that is related to the movie world, but the local CSV dataset did not provide a reliable direct answer.
Answer from general movie knowledge. Use Google Search grounding when available.

Conversation context:
{conversation_context or "none"}

Current user question:
{user_message}

Rules:
1. Answer ONLY the user's actual question. Do not turn it into a new recommendation flow.
2. If the question asks who acts/plays/stars/participates in a movie, provide the main cast.
3. If the question asks about plot, director, release year, age rating, sequels, genre, or similar factual movie information, answer that directly.
4. If the movie title is in Hebrew, infer the known English/international title when reasonable.
5. If there are multiple versions of a movie with the same translated title, mention the likely version and keep the answer cautious.
6. Do not ask whether the user wants home or cinema unless the user is clearly asking for a recommendation.
7. Answer naturally in Hebrew and do not expose implementation details.
""".strip()
    return query_gemini_with_web_search(prompt)

def build_cinema_search_prompt(full_request: str) -> str:
    """Prompt used only when the answer needs current cinema information."""
    return f"""
You are a Hebrew movie recommendation agent with Google Search grounding.
The user asks for CURRENT cinema / now playing recommendations. Use Google Search when available.
Prioritize public Israeli cinema sources such as Cinema City and YES Planet when relevant, and use them to infer currently playing films. If the user did not provide a city or branch, give general current options and suggest checking the specific branch before ordering tickets.

Full user request and conversation context:
{full_request}

Structured hard constraints detected from the full context:
{summarize_detected_constraints_for_prompt(full_request)}

Very important:
1. First infer the FULL intent from the whole context, including typos and follow-up messages. The latest message may be only "בבית" or "בקולנוע", so preserve the previous request exactly.
2. Treat constraints as HARD filters, not suggestions. Recommend a movie only if it matches ALL requested constraints.
3. Preserve the requested genre:
   - Comedy/מצחיק/קומדיה: recommend ONLY comedy or clearly comic/parody movies. Do NOT include action, thriller, drama, fantasy, family or animation unless the movie is clearly also a comedy.
   - Thriller/מתח: recommend ONLY thriller/suspense movies or movies whose description clearly matches suspense/thriller.
   - Horror/אימה/עימה: recommend ONLY horror or clearly horror-adjacent movies.
4. Preserve runtime constraints from the full context. If the user asked for under two hours / less than 120 minutes / up to 90 minutes, do NOT recommend movies longer than that unless you clearly say you could not verify enough matching options.
5. Preserve age constraints. If the user asks for under 18 / children / family suitability, mention age suitability carefully and avoid adult-only content.
6. If you cannot verify enough currently playing movies that match ALL constraints, say that clearly and return fewer options. Do NOT fill the list with unrelated movies just to reach 5 results. If the requested genre is Horror, every returned title must be horror or clearly horror-adjacent; if not, do not include it. Never include family/fantasy/drama titles as Horror unless the source clearly classifies them as horror or horror-comedy.
7. Prioritize Cinema City and YES Planet public pages when relevant. If no city/branch was provided, give general current options and say that exact showtimes depend on the branch.
8. Answer in natural Hebrew for the end user. Do NOT use headings like "למה מתאים", "למה זה מתאים לבקשה", or any internal reasoning.
9. For each movie use this friendly format:
🎬 Movie name
תקציר קצר: one or two friendly sentences.
סגנון: genre/style.
הערת גיל/זמינות: only if relevant.
10. Add a short note that showtimes and availability depend on the local cinema.
11. End with a friendly follow-up question: "רוצה שאפרט על אחד הסרטים עם שחקנים, מגבלת גיל, שנת יציאה או סרטי המשך?"
""".strip()


# ----------------------------------------------------------------------
# 8. Agent routing logic
# ----------------------------------------------------------------------

def movie_agent(user_message: str, session_id: str = "default") -> str:
    original_message = user_message.strip()
    msg = normalize_user_text(original_message).lower().strip()

    if not msg:
        return "כתבי לי איזה סרט בא לך ואנסה להמליץ 😊"

    summary_followup = answer_summary_followup(original_message, session_id)
    if summary_followup:
        return summary_followup

    if msg in ["hi", "hello", "hey", "שלום", "היי", "הי"]:
        local = (
            "היי! אני סוכן AI להמלצות סרטים 🎬\n"
            "אני עובד בשני שלבים: קודם מחפש בדאטה המקומי שלנו מתוך קובץ ה-CSV, "
            "ורק אם מדובר במידע שלא קיים שם, למשל סרטים שמוקרנים עכשיו בקולנוע, אני נעזר ב-Gemini לחיפוש עדכני.\n\n"
            "אפשר לכתוב למשל:\n"
            "- בא לי סרט אקשן\n"
            "- סרט דומה לספיידרמן\n"
            "- איזה סרטי קומדיה מוקרנים עכשיו בקולנוע?\n"
            "- show trends\n"
            "- סרט דומה לספיידרמן"
        )
        return final_answer(original_message, local, "greeting", "data_found")

    # First let Gemini classify the actual task.
    # A sentence can contain the word "movie/סרט" but still be an information
    # question, not a recommendation request.
    context_for_router = build_context_for_router(session_id)
    routing = get_routing_analysis(original_message, conversation_context=context_for_router)

    if routing.get("detail_question") or routing.get("user_task") == "movie_details":
        local, status = get_movie_info(original_message)
        return final_answer(
            original_message,
            local,
            "movie information with Gemini fallback",
            status,
            allow_general_knowledge=True,
        )

    # If the user asks for a totally generic recommendation, do not return arbitrary top movies.
    # Ask for a few preferences first.
    if should_ask_for_recommendation_details(original_message):
        return build_recommendation_details_question()

    # Detail follow-ups must be handled BEFORE the home/cinema question flow.
    # Otherwise a sentence like "מי השחקנים בסרט חוות החיות" is misread as a
    # fresh recommendation request only because it contains the word "סרט".
    if is_movie_detail_followup(original_message) and build_context_for_router(session_id):
        local, status = get_movie_info(original_message)
        if status == "data_found":
            return final_answer(original_message, local, "contextual movie information from local CSV", status)
        return query_gemini_with_web_search(
            build_movie_detail_search_prompt(original_message, build_context_for_router(session_id))
        )

    # If the agent previously asked "home or cinema", use Gemini to understand the user's answer in context.
    # This handles typos like "קוללנוע" through Gemini routing, not hard-coded typo lists.
    pending_viewing_request = pop_viewing_mode_pending(session_id)
    if pending_viewing_request:
        combined_request = f"{pending_viewing_request}. Viewing preference / follow-up: {original_message}"
        context = f"Previous movie request: {pending_viewing_request}. The agent asked the user to choose home viewing or current cinema screenings."
        answer_routing = get_routing_analysis(original_message, conversation_context=context)
        preference = answer_routing.get("viewing_preference", "unclear")

        if preference == "cinema" or answer_routing.get("needs_current_external_info"):
            normalized_answer = answer_routing.get("normalized_query_hebrew") or original_message
            full_cinema_request = f"{pending_viewing_request}. Viewing preference: cinema. User follow-up: {normalized_answer}"
            return answer_cinema_request(session_id, full_cinema_request)

        if preference == "home":
            # Preserve the original pending request explicitly. The router may normalize
            # the short answer "בבית" but accidentally drop constraints such as actor
            # names. The original request is the source of truth; the router output is
            # only an optional cleaned follow-up.
            normalized_answer = answer_routing.get("normalized_query_hebrew") or original_message
            full_home_request = f"{pending_viewing_request}. Viewing preference: home. User follow-up: {normalized_answer}"
            save_last_home_request(session_id, full_home_request)
            local, status = recommend_by_text(full_home_request)
            if status == "data_found":
                save_last_recommendations(session_id, local)
            return final_answer(full_home_request, local, "home viewing recommendation from local CSV", status, allow_general_knowledge=(status == "data_missing"))

        # If the user answered with extra constraints instead of explicitly choosing,
        # keep the original request and ask again without treating it as unrelated.
        save_viewing_mode_pending(session_id, combined_request)
        return "רק כדי לדייק — את רוצה המלצה לצפייה בבית או לבדוק סרטים שמוקרנים עכשיו בקולנוע?"

    # Before asking "home or cinema", let Gemini decide whether the user is really asking
    # for a recommendation or asking a factual movie question.
    # This avoids hard-coding every Hebrew wording such as "who plays", "who stars", etc.
    if routing.get("is_movie_related") and not routing.get("recommendation_request") and routing.get("user_task") not in ["recommendation", "similar_movies", "data_analysis"]:
        return answer_movie_related_with_gemini(original_message, context_for_router)

    # Broad local recommendation requests should NOT depend on Gemini routing.
    # Example: "סרט מתח" / "בא לי סרט אימה".
    # The agent first asks home vs cinema and stores the context.
    # Only after the user chooses cinema do we use Gemini for current screenings.
    if should_ask_home_or_cinema(original_message):
        save_viewing_mode_pending(session_id, original_message)
        save_last_recommendation_request(session_id, original_message)
        return build_home_or_cinema_question(original_message)

    # Gemini already understood the user's intent above.
    # The actual answer still uses the CSV first, except for current external info like cinema screenings.

    # Universal context handling: if the user answers "home"/"cinema" or adds details
    # after any previous movie request, keep ALL earlier constraints. This is not limited
    # to cinema. The active context is always the latest movie-related request.
    active_context = get_last_active_context(session_id)
    if active_context:
        preference = routing.get("viewing_preference", "unclear")
        normalized_current = routing.get("normalized_query_hebrew") or original_message
        if preference == "home":
            full_home_request = f"{active_context}. Viewing preference: home. User follow-up: {normalized_current}"
            save_last_home_request(session_id, full_home_request)
            local, status = recommend_by_text(full_home_request)
            if status == "data_found":
                save_last_recommendations(session_id, local)
            return final_answer(full_home_request, local, "contextual home recommendation from local CSV", status, allow_general_knowledge=(status == "data_missing"))
        if preference == "cinema" or routing.get("needs_current_external_info"):
            full_cinema_request = f"{active_context}. Viewing preference: cinema. User follow-up: {normalized_current}"
            return answer_cinema_request(session_id, full_cinema_request)
        if is_contextual_refinement(original_message):
            mode = get_last_viewing_mode(session_id)
            refined_request = f"{active_context}. Additional user requirement: {original_message}"
            if mode == "cinema":
                return answer_cinema_request(session_id, refined_request)
            save_last_home_request(session_id, refined_request)
            local, status = recommend_by_text(refined_request)
            if status == "data_found":
                save_last_recommendations(session_id, local)
            return final_answer(refined_request, local, "contextual refined home recommendation from local CSV", status, allow_general_knowledge=(status == "data_missing"))

    if not routing.get("is_movie_related"):
        # If this looks like a short follow-up in an existing movie conversation, do not reject it.
        # Ask Gemini with the previous context first; if it identifies cinema/current info, continue accordingly.
        if context_for_router and is_short_followup_answer(original_message):
            contextual_routing = get_routing_analysis(original_message, conversation_context=context_for_router)
            if contextual_routing.get("viewing_preference") == "cinema" or contextual_routing.get("needs_current_external_info"):
                previous = get_last_recommendation_request(session_id) or get_last_cinema_request(session_id)
                combined = f"{previous}. User follow-up: {original_message}"
                return answer_cinema_request(session_id, combined)
            return "אני מבינה שזה המשך לשיחה על סרטים 😊 את רוצה שאבדוק סרטים בקולנוע או שאמליץ מהדאטה שלנו לצפייה בבית?"
        return unrelated_response(original_message)

    # If the user asks for details about a movie from the previous cinema/current results
    # (for example: "מי השחקנים בתגלית"), this information may not exist in the CSV.
    # Use Gemini Search with the current conversation context instead of falling back to generic CSV recommendations.
    if is_movie_detail_followup(original_message) and build_context_for_router(session_id):
        # Detail follow-ups use the whole conversation context, no matter whether the
        # previous recommendation was for home or cinema. Try the CSV first when a
        # clear title exists; if not, Gemini completes the missing information.
        local, status = get_movie_info(original_message)
        if status == "data_found":
            return final_answer(original_message, local, "contextual movie information from local CSV", status)
        return query_gemini_with_web_search(
            build_movie_detail_search_prompt(original_message, build_context_for_router(session_id))
        )

    # If the previous flow was cinema/current movies and the user now adds a constraint
    # like "מתאים לילדים מתחת לגיל 18", keep the conversation context and search again.
    last_cinema_request = get_last_cinema_request(session_id)
    if last_cinema_request and get_last_viewing_mode(session_id) == "cinema" and is_contextual_refinement(original_message):
        combined_cinema_request = f"{last_cinema_request}. Additional user requirement: {original_message}"
        return answer_cinema_request(session_id, combined_cinema_request)

    # Requests about currently playing movies / cinema screenings are not in the historical CSV.
    # Route them directly to Gemini with Google Search instead of returning old CSV movies.
    if routing.get("needs_current_external_info"):
        previous_request = get_last_recommendation_request(session_id)
        normalized_request = routing.get("normalized_query_hebrew") or original_message
        cinema_request = normalized_request
        if previous_request and len(original_message.split()) <= 5:
            cinema_request = f"{previous_request}. Viewing preference: {normalized_request}"
        return answer_cinema_request(session_id, cinema_request)

    if any(term in msg for term in ["cluster", "clusters", "clustering", "קלאסטר", "אשכול"]):
        local, status = describe_clusters()
        return final_answer(original_message, local, "clustering", status)

    if any(term in msg for term in ["trend", "trends", "popular", "pattern", "מגמות", "דפוסים", "פופולרי"]):
        local, status = show_trends()
        return final_answer(original_message, local, "trend analysis", status)

    # Similar movie requests should always try the CSV first.
    if any(term in msg for term in ["similar to", "movies like", "movie like", "דומה ל", "דומים ל", "כמו", "בסגנון"]):
        local, status = recommend_by_movie(find_movie_title_in_message(original_message))
        if status == "data_found":
            save_last_recommendations(session_id, local)
        return final_answer(original_message, local, "similarity recommendation from local CSV", status, allow_general_knowledge=(status == "data_missing"))

    if any(term in msg for term in ["runtime", "length", "duration", "minutes", "short", "long", "דקות", "קצר", "ארוך", "מתחת", "פחות", "עד", "מעל", "שעה", "שעתיים", "שעתים", "שעות"]):
        has_filter_word = any(term in msg for term in ["under", "less than", "over", "more than", "short", "long", "מתחת", "פחות", "עד", "מעל", "קצר", "ארוך"])
        if has_filter_word:
            # If the message is a recommendation request with runtime + genre/style,
            # use the full recommendation function so we do NOT lose filters like
            # Comedy/Horror/under-18. A pure runtime request can still use
            # recommend_by_runtime.
            if has_recommendation_intent(original_message) or detect_requested_genres(original_message):
                save_last_recommendation_request(session_id, original_message)
                local, status = recommend_by_text(original_message)
                if status == "data_found":
                    save_last_recommendations(session_id, local)
                return final_answer(original_message, local, "filtered recommendation with runtime from local CSV", status, allow_general_knowledge=(status == "data_missing"))

            local, status = recommend_by_runtime(original_message)
            if status == "data_found":
                save_last_recommendations(session_id, local)
            return final_answer(original_message, local, "runtime recommendation from local CSV", status, allow_general_knowledge=(status == "data_missing"))
        local, status = get_movie_info(original_message)
        return final_answer(original_message, local, "movie runtime information from local CSV", status, allow_general_knowledge=(status == "data_missing"))

    if any(term in msg for term in ["review", "reviews", "rating", "ratings", "who directed", "director", "plot", "about", "cast", "summary", "synopsis", "תקציר", "ביקורות", "ביקורת", "דירוג", "במאי", "עלילה", "שחקנים", "מי השחקנים", "סרטי המשך", "שנת יציאה"]):
        local, status = get_movie_info(original_message)
        return final_answer(original_message, local, "movie information with Gemini fallback", status, allow_general_knowledge=True)

    # Default: local CSV recommendation first. Gemini only if the local dataset cannot answer.
    if has_recommendation_intent(original_message):
        save_last_recommendation_request(session_id, original_message)
    local, status = recommend_by_text(original_message)
    if status == "data_found":
        save_last_recommendations(session_id, local)
        return final_answer(original_message, local, "free text recommendation from local CSV", status)

    return final_answer(original_message, local, "movie fallback", "data_missing", allow_general_knowledge=True)

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
    try:
        reply = movie_agent(user_message, session_id=session_id)
    except Exception as exc:
        # Never let the browser show a generic fetch error.
        # The real error is printed in Render Logs for debugging.
        print(f"Chat route error: {exc}")
        reply = (
            "נתקלתי בתקלה רגעית בעיבוד הבקשה. "
            "נסי לנסח שוב, למשל: 'סרט מתח', 'בקולנוע', או 'בבית'."
        )
    return jsonify({"reply": reply})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "movies_loaded": len(df),
        "clusters": int(n_clusters),
        "gemini_configured": bool(get_valid_gemini_api_key()) and genai is not None,
        "gemini_models": model_fallback_list(),
    })


if __name__ == "__main__":
    print(f"Loaded {len(df)} movies")
    print(f"Clusters: {n_clusters}")
    print("Gemini configured:", bool(get_valid_gemini_api_key()) and genai is not None)
    if os.getenv("GEMINI_API_KEY") and not get_valid_gemini_api_key():
        print("Gemini API key was ignored because it is missing/invalid.")
    app.run(host="127.0.0.1", port=5000, debug=True)    
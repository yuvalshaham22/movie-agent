# -*- coding: utf-8 -*-
"""
Movie Recommendation AI Agent
-----------------------------
Stable final version for the AI/ML project.

Main behavior:
1. Understands a free text movie request in Hebrew/English.
2. Saves the user's filters while asking whether the user wants home viewing or cinema.
3. After the user chooses home/cinema, it keeps the original filters.
4. Searches the local CSV first for every movie-related request.
5. Uses Gemini only as an internal fallback when the local CSV cannot answer.
6. Does not expose to the user whether the answer came from CSV or Gemini.
"""

import os
import sys
import ast
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

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

app = Flask(__name__)
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
# Data loading
# ----------------------------------------------------------------------

def extract_names_from_json(value: object, job_filter: Optional[str] = None) -> List[str]:
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
            f"Missing {path}. Put movies_agent_clean_data_small.csv in the project folder."
        )

    data = pd.read_csv(path)

    for col in ["combined_text", "genres_movielens", "title_movielens", "overview"]:
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = 0
        data[col] = pd.to_numeric(data[col], errors="coerce")
        median_value = data[col].median()
        if pd.isna(median_value):
            median_value = 0
        data[col] = data[col].fillna(median_value)

    if "age_limit" not in data.columns:
        data["age_limit"] = 13
    data["age_limit"] = pd.to_numeric(data["age_limit"], errors="coerce").fillna(13).astype(int)

    if "age_restriction_label" not in data.columns:
        data["age_restriction_label"] = data["age_limit"].apply(
            lambda x: "לכל המשפחה" if int(x) <= 0 else f"{int(x)}+"
        )
    data["age_restriction_label"] = data["age_restriction_label"].fillna("13+").astype(str)

    if "suitable_under_18" not in data.columns:
        data["suitable_under_18"] = data["age_limit"] < 18

    if "cast_names" in data.columns:
        data["cast_names"] = data["cast_names"].fillna("").astype(str)
    elif "cast" in data.columns:
        data["cast_names"] = data["cast"].apply(lambda value: ", ".join(extract_names_from_json(value)))
    else:
        data["cast_names"] = ""

    if "director_names" in data.columns:
        data["director_names"] = data["director_names"].fillna("").astype(str)
    elif "crew" in data.columns:
        data["director_names"] = data["crew"].apply(lambda value: ", ".join(extract_names_from_json(value, "Director")))
    else:
        data["director_names"] = ""

    data = data.drop_duplicates(subset=["title_movielens"]).copy()
    data = data[data["title_movielens"].str.strip() != ""]
    data = data.sort_values(by=["vote_count", "popularity"], ascending=False)
    data = data.head(MAX_MOVIES_FOR_APP).reset_index(drop=True)

    data["title_lower"] = data["title_movielens"].str.lower()
    data["title_no_year"] = data["title_lower"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()
    data["genres_list"] = data["genres_movielens"].fillna("").astype(str).str.split("|")
    data["cast_names_lower"] = data["cast_names"].str.lower()
    data["director_names_lower"] = data["director_names"].str.lower()
    return data


df = load_movies(DATA_PATH)
all_titles_no_year = df["title_no_year"].astype(str).tolist()

# ----------------------------------------------------------------------
# ML setup
# ----------------------------------------------------------------------

vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(df[NUMERIC_FEATURES])

# Keep clustering lightweight so the Flask app starts quickly on Render.
n_clusters = min(8, max(2, len(df) // 400))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(numeric_matrix)

anomaly_model = IsolationForest(contamination=0.03, random_state=42)
df["anomaly_score"] = anomaly_model.fit_predict(numeric_matrix)
df["is_anomaly"] = df["anomaly_score"] == -1

# ----------------------------------------------------------------------
# Conversation state
# ----------------------------------------------------------------------

CONVERSATION_STATE: Dict[str, Dict[str, str]] = {}
INTENT_NORMALIZATION_CACHE: Dict[str, str] = {}
VIEWING_INTENT_CACHE: Dict[str, str] = {}


def get_state(session_id: str) -> Dict[str, str]:
    if session_id not in CONVERSATION_STATE:
        CONVERSATION_STATE[session_id] = {}
    return CONVERSATION_STATE[session_id]


def save_pending_request(session_id: str, user_text: str) -> None:
    state = get_state(session_id)
    state["pending_recommendation_request"] = user_text
    state["pending_needs_genre"] = "0" if has_genre_or_style_preferences(user_text) else "1"
    state["last_movie_context"] = user_text


def pop_pending_request(session_id: str) -> Optional[str]:
    return get_state(session_id).pop("pending_recommendation_request", None)


def ask_genre_question(session_id: str, user_text: str, viewing_mode: str) -> str:
    state = get_state(session_id)
    state["pending_recommendation_request"] = user_text
    state["pending_viewing_mode"] = viewing_mode
    state["pending_needs_genre"] = "1"
    state["last_movie_context"] = user_text
    state["last_viewing_mode"] = viewing_mode
    location = "בבית" if viewing_mode == "home" else "בקולנוע"
    return (
        f"מעולה, נלך על צפייה {location} 😊\n"
        "איזה ז׳אנר או סגנון בא לך לראות? למשל אקשן, קומדיה, מתח, רומנטי או משהו קליל."
    )


def save_last_context(session_id: str, user_text: str, viewing_mode: str = "") -> None:
    state = get_state(session_id)
    # Viewing mode is stored separately so changing from home to cinema later
    # does not leave conflicting instructions inside the movie preferences.
    clean_context = re.sub(r"\.?\s*בחירת צפייה:\s*(?:home|cinema)\.?", "", user_text, flags=re.IGNORECASE).strip()
    state["last_movie_context"] = clean_context
    if viewing_mode:
        state["last_viewing_mode"] = viewing_mode


def get_last_context(session_id: str) -> str:
    return get_state(session_id).get("last_movie_context", "")


def get_last_viewing_mode(session_id: str) -> str:
    return get_state(session_id).get("last_viewing_mode", "")


def save_focused_movie(session_id: str, movie_title: str) -> None:
    if movie_title:
        get_state(session_id)["focused_movie_title"] = movie_title


def get_focused_movie(session_id: str) -> str:
    return get_state(session_id).get("focused_movie_title", "")


def merge_with_movie_context(session_id: str, user_text: str) -> str:
    previous_context = get_last_context(session_id)
    if not previous_context:
        return user_text
    return f"{previous_context}. דרישה נוספת מהמשתמש: {user_text}"


def has_concrete_movie_preferences(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    mood_or_style_terms = [
        "קליל", "כבד", "מרגש", "מצחיק", "מפחיד", "מותח", "אפל", "רומנטי",
        "לכל המשפחה", "עם הילדים", "עם הבן", "עם הבת", "דייט",
        "light", "emotional", "funny", "scary", "dark", "romantic", "family",
    ]
    return bool(
        detect_requested_genres(text)
        or extract_requested_actor(text)
        or extract_runtime_filter(text) != (0, 1000)
        or extract_min_rating(text) is not None
        or extract_age_filter(text)[:2] != (None, None)
        or any(term in text for term in mood_or_style_terms)
    )


def has_genre_or_style_preferences(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    style_terms = [
        "קליל", "כבד", "מרגש", "מצחיק", "מפחיד", "מותח", "אפל", "רומנטי",
        "light", "emotional", "funny", "scary", "dark", "romantic",
    ]
    return bool(detect_requested_genres(text) or any(term in text for term in style_terms))


def is_recommendation_followup_filter(user_text: str) -> bool:
    """True for constraints that naturally refine the active recommendation."""
    return bool(
        extract_runtime_filter(user_text) != (0, 1000)
        or extract_min_rating(user_text) is not None
        or extract_age_filter(user_text)[:2] != (None, None)
        or extract_requested_actor(user_text)
    )


def is_collection_recommendation_filter(user_text: str) -> bool:
    """Distinguish list filtering from a detail question about one movie."""
    text = normalize_user_text(user_text).lower()
    collection_terms = [
        "סרטים", "תמליץ", "תמליצי", "להמליץ", "המלצות",
        "movies", "films", "recommend", "recommendations",
    ]
    return is_recommendation_followup_filter(text) and any(term in text for term in collection_terms)


def is_genre_or_style_only(user_text: str) -> bool:
    """True when the user supplied a genre/style but did not start a new request."""
    text = normalize_user_text(user_text).lower()
    request_terms = [
        "סרט", "המלצה", "תמליץ", "רוצה", "אשמח", "בא לי",
        "movie", "film", "recommend", "want",
    ]
    return has_genre_or_style_preferences(text) and not any(term in text for term in request_terms)


def ask_cinema_preferences(session_id: str, user_text: str) -> str:
    save_last_context(session_id, user_text, "cinema")
    get_state(session_id)["waiting_for_cinema_preferences"] = "1"
    return (
        "בשמחה, נחפש סרט שמוקרן בקולנוע 🎬\n"
        "כדי שאוכל לחפש באינטרנט משהו שבאמת יתאים לך, מה ההעדפה שלך?\n\n"
        "אפשר לכתוב ז׳אנר או סגנון כמו קומדיה, אקשן, קליל או מרגש; "
        "שחקן או שחקנית שאוהבים; למי הסרט צריך להתאים; או סרט שאהבת ורוצה משהו דומה לו."
    )

# ----------------------------------------------------------------------
# NLP helpers
# ----------------------------------------------------------------------

GENRE_MAP: Dict[str, List[str]] = {
    "Action": ["action", "אקשן", "פעולה"],
    "Adventure": ["adventure", "הרפתקה", "הרפתקאות", "הרפתקני"],
    "Animation": ["animation", "animated", "cartoon", "אנימציה", "מצויר", "מצוייר"],
    "Children": ["children", "kids", "kid", "family", "ילדים", "לילדים", "משפחה", "משפחתי"],
    "Comedy": ["comedy", "comedies", "funny", "humor", "laugh", "מצחיק", "קומדיה", "קומדייה", "קומדי"],
    "Crime": ["crime", "פשע"],
    "Documentary": ["documentary", "דוקומנטרי", "תיעודי"],
    "Drama": ["drama", "דרמה", "מרגש"],
    "Fantasy": ["fantasy", "פנטזיה", "קסם"],
    "Horror": ["horror", "scary", "ghost", "אימה", "עימה", "איימה", "מפחיד"],
    "Mystery": ["mystery", "מסתורין", "תעלומה"],
    "Romance": ["romance", "romantic", "love", "רומנטי", "רומנטיקה", "אהבה"],
    "Sci-Fi": ["sci-fi", "scifi", "science fiction", "מדע בדיוני", "חלל", "עתידני"],
    "Thriller": ["thriller", "suspense", "מתח", "מותחן"],
    "War": ["war", "מלחמה"],
    "Musical": ["musical", "music", "מוזיקלי", "מוזיקה"],
    "IMAX": ["imax"],
}

ACTOR_ALIASES = {
    "אדם סנדלר": "Adam Sandler",
    "אדאם סנדלר": "Adam Sandler",
    "adam sandler": "Adam Sandler",
    "בראד פיט": "Brad Pitt",
    "בארד פיט": "Brad Pitt",
    "brad pitt": "Brad Pitt",
    "brad pit": "Brad Pitt",
    "ג'ניפר לורנס": "Jennifer Lawrence",
    "ג׳ניפר לורנס": "Jennifer Lawrence",
    "jennifer lawrence": "Jennifer Lawrence",
    "טום הנקס": "Tom Hanks",
    "tom hanks": "Tom Hanks",
    "ליאונרדו דיקפריו": "Leonardo DiCaprio",
    "לאונרדו דיקפריו": "Leonardo DiCaprio",
    "leonardo dicaprio": "Leonardo DiCaprio",
    "ג'וני דפ": "Johnny Depp",
    "ג׳וני דפ": "Johnny Depp",
    "johnny depp": "Johnny Depp",
}

MOVIE_TITLE_ALIASES = {
    "בחזרה לעתיד": "Back to the Future",
    "ספיידרמן": "Spider-Man",
    "ספיידר מן": "Spider-Man",
    "באטמן": "Batman",
    "האביר האפל": "Dark Knight",
    "טיטאניק": "Titanic",
    "אווטאר": "Avatar",
    "צעצוע של סיפור": "Toy Story",
    "הארי פוטר": "Harry Potter",
    "שרק": "Shrek",
    "מטריקס": "Matrix",
    "הנוקמים": "Avengers",
    "מלך האריות": "Lion King",
    "לשבור את הקרח": "Frozen",
    "פארק היורה": "Jurassic Park",
    "מלחמת הכוכבים": "Star Wars",
    "פורסט גאמפ": "Forrest Gump",
}

KNOWN_SEQUEL_NOTES = {
    "home": (
        "ל-Home (2015) אין סרט קולנוע המשך רשמי. "
        "קיימת סדרת המשך טלוויזיונית בשם Home: Adventures with Tip & Oh."
    ),
}

MOVIE_DOMAIN_TERMS = {
    "movie", "film", "movies", "films", "cinema", "actor", "actress", "director", "cast",
    "genre", "runtime", "rating", "plot", "review",
    "סרט", "סרטים", "קולנוע", "שחקן", "שחקנית", "שחקנים", "במאי", "קאסט", "זאנר", "ז׳אנר",
    "אורך", "דקות", "דירוג", "עלילה", "תקציר", "ביקורת",
}

NON_MOVIE_DOMAIN_TERMS = {
    "עוגה", "עוגות", "אוכל", "מתכון", "מתכונים", "מסעדה", "מסעדות",
    "cake", "cakes", "food", "recipe", "recipes", "restaurant", "restaurants",
}


def normalize_user_text(text: str) -> str:
    updated = str(text or "")
    for hebrew_title, english_title in MOVIE_TITLE_ALIASES.items():
        updated = re.sub(re.escape(hebrew_title), english_title, updated, flags=re.IGNORECASE)
    return updated


def detect_requested_genres(user_text: str) -> List[str]:
    text = normalize_user_text(user_text).lower()
    detected: List[str] = []
    for genre, keywords in GENRE_MAP.items():
        for keyword in keywords:
            pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
            if re.search(pattern, text):
                detected.append(genre)
                break
    # If the user asks for children/family comedy, keep both filters.
    return detected


def extract_requested_actor(user_text: str) -> Optional[str]:
    text = normalize_user_text(user_text).lower()

    for alias, actor in ACTOR_ALIASES.items():
        if alias.lower() in text:
            return actor

    # English: with Adam Sandler / actor Adam Sandler / starring Adam Sandler
    english_match = re.search(
        r"(?:with|actor|actress|starring|starred by)\s+([a-zA-Z][a-zA-Z'\-]+(?:\s+[a-zA-Z][a-zA-Z'\-]+){0,3})",
        text,
        flags=re.IGNORECASE,
    )
    if english_match:
        return english_match.group(1).strip().title()

    # Hebrew actor requests. A generic "עם" is intentionally excluded so
    # phrases such as "עם הבן שלי" are not mistaken for an actor name.
    hebrew_match = re.search(
        r"(?:שחקן|שחקנית|בכיכוב|בכיכובה|בכיכובו)\s+([א-ת׳'\-]+(?:\s+[א-ת׳'\-]+){0,3})",
        text,
    )
    if hebrew_match:
        candidate = hebrew_match.group(1).strip(" .,!?:;\"'")
        candidate = re.split(r"\s+(?:באורך|בזאנר|בז׳אנר|ודירוג|דירוג|מעל|מתחת|עד|של|בבית|בקולנוע)", candidate)[0].strip()
        for alias, actor in ACTOR_ALIASES.items():
            if candidate and (candidate in alias.lower() or alias.lower() in candidate):
                return actor
        return candidate

    return None


def extract_runtime_filter(text: str) -> Tuple[int, int]:
    text = normalize_user_text(text).lower()
    min_runtime, max_runtime = 0, 1000
    runtime_context_terms = ["אורך", "דקות", "דקה", "minutes", "minute", "runtime", "length", "duration"]
    has_runtime_context = any(term in text for term in runtime_context_terms)

    under_terms = ["under", "less than", "עד", "פחות", "מתחת", "לכל היותר", "מקסימום"]
    over_terms = ["over", "more than", "מעל", "יותר", "לפחות", "מינימום"]
    has_under = any(term in text for term in under_terms)
    has_over = any(term in text for term in over_terms)

    if "שעה וחצי" in text or "1.5 שעות" in text or "1.5 hour" in text:
        if has_over:
            return 90, 1000
        if has_under:
            return 0, 90
        return 80, 110

    if "שעתיים" in text or "שעתים" in text:
        if has_over:
            return 120, 1000
        if has_under:
            return 0, 120
        return 100, 140

    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:שעות|שעה|hours?|hrs?)", text)
    if hour_match:
        target = int(float(hour_match.group(1)) * 60)
        if has_over:
            return target, 1000
        if has_under:
            return 0, target
        return max(0, target - 20), target + 20

    numbers = [int(x) for x in re.findall(r"\d+", text)]
    if "קצר" in text or "short" in text:
        max_runtime = 90
    elif "ארוך" in text or "long" in text:
        min_runtime = 120
    elif has_runtime_context and has_under and numbers:
        max_runtime = numbers[0]
    elif has_runtime_context and has_over and numbers:
        min_runtime = numbers[0]
    elif numbers and has_runtime_context:
        target = numbers[0]
        min_runtime = max(0, target - 20)
        max_runtime = target + 20

    return min_runtime, max_runtime


def extract_min_rating(text: str) -> Optional[float]:
    text = normalize_user_text(text).lower()
    if any(term in text for term in ["דירוג", "rating", "rated", "ציון"]):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
        valid = [x for x in nums if 0 <= x <= 10]
        if valid:
            return max(valid)
        if any(term in text for term in ["גבוה", "high", "טוב", "מעולה"]):
            return 7.0
        return 6.8
    if "דירוג גבוה" in text or "highly rated" in text:
        return 7.0
    return None


def extract_age_filter(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    text = normalize_user_text(text).lower()
    min_age = None
    max_age = None
    description = None

    age_context_terms = [
        "גיל", "ילדים", "לילדים", "נוער", "מתבגרים", "משפחה", "משפחתי", "מתאים",
        "age", "kids", "children", "family", "teen", "teens"
    ]
    has_age_context = any(term in text for term in age_context_terms)

    if re.search(r"מתחת\s*(?:לגיל|ל)?\s*18", text) or "under 18" in text or "לא 18+" in text:
        max_age = 17
        description = "מתאים מתחת לגיל 18"
    elif any(term in text for term in ["ילדים", "לילדים", "kids", "children", "family", "משפחה", "משפחתי"]):
        max_age = 13
        description = "מתאים לילדים/משפחה"
    elif any(term in text for term in ["נוער", "מתבגרים", "teen", "teens"]):
        max_age = 16
        description = "מתאים לנוער"

    # Numeric age filters are applied only when the user explicitly talks about age.
    # This prevents "מעל 90 דקות" from being misread as "מעל גיל 90".
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

    return min_age, max_age, description


def detect_viewing_preference(text: str) -> str:
    text = normalize_user_text(text).lower().strip()
    # Current user answer gets priority over previous context.
    if re.search(r"\b(home|streaming)\b", text) or any(term in text for term in ["בבית", "בית", "סטרימינג", "טלויזיה", "טלוויזיה"]):
        return "home"
    if re.search(r"\b(cinema|theater|theatre|now playing)\b", text) or any(term in text for term in ["בקולנוע", "קולנוע", "הקרנה", "מוקרנים", "מוקרן"]):
        return "cinema"
    return "unclear"


def is_movie_related(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    tokens = set(re.findall(r"[a-zA-Zא-ת0-9+׳'-]+", text))
    has_domain_term = bool(tokens.intersection(MOVIE_DOMAIN_TERMS))
    if not has_domain_term:
        hebrew_prefixes = ("ב", "ל", "ה", "ו", "כ", "מ", "ש")
        has_domain_term = any(
            len(token) > 2 and token.startswith(hebrew_prefixes) and token[1:] in MOVIE_DOMAIN_TERMS
            for token in tokens
        )
    return bool(has_domain_term or detect_requested_genres(text) or find_title_in_message(text))


def is_explicitly_non_movie_related(user_text: str) -> bool:
    text = str(user_text or "").lower()
    tokens = set(re.findall(r"[a-zA-Zא-ת0-9+'-]+", text))
    return bool(tokens.intersection(NON_MOVIE_DOMAIN_TERMS)) and not is_movie_related(user_text)


def has_recommendation_intent(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    if is_explicitly_non_movie_related(user_text):
        return False
    if detect_requested_genres(text) or extract_runtime_filter(text) != (0, 1000):
        return True
    if extract_requested_actor(text) and any(term in text for term in [
        "סרט", "תן לי", "תני לי", "המלצה", "רוצה", "בא לי", "אשמח",
        "movie", "film", "recommend", "want",
    ]):
        return True
    if any(term in str(user_text).lower() for term in ["תפתיע אותי", "surprise me"]):
        return True
    recommendation_terms = [
        "המלצה", "תמליץ", "תמליצי", "בא לי", "רוצה", "אשמח",
        "recommend", "recommendation", "want", "looking for"
    ]
    return is_movie_related(user_text) and any(term in text for term in recommendation_terms)


def is_detail_question(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    return any(term in text for term in [
        "מי משחק", "מי השחקנים", "שחקן ראשי", "השחקן הראשי", "שחקנית ראשית", "השחקנית הראשית",
        "שחקנים", "קאסט", "cast", "actor", "actors",
        "מי ביים", "במאי", "director", "על מה", "תקציר", "פירוט", "לפרט", "תפרט", "תפרטי",
        "ספר לי על", "ספרי לי על", "plot", "about", "שנת יציאה", "release",
        "דירוג", "rating", "ביקורת", "review", "אורך", "runtime", "כמה זמן",
        "סרט המשך", "סרטי המשך", "המשך של", "sequel", "sequels",
        "מתאים לילדים", "מתאים לילד", "מתאים לגיל", "מתחת לגיל", "מגבלת גיל",
    ])


def refers_to_focused_movie(user_text: str) -> bool:
    text = normalize_user_text(user_text).lower()
    return any(term in text for term in [
        "הסרט", "לסרט", "בסרט", "של הסרט", "אותו", "עליו", "על הסרט",
        "זה מתאים", "הוא מתאים", "היא מתאימה", "מתאים לילדים", "מתאים לילד",
        "מתאים לגיל", "מתחת לגיל", "מגבלת הגיל שלו", "מגבלת הגיל שלה",
        "the movie", "this movie", "the film", "it",
    ])


def find_title_in_message(message: str) -> Optional[str]:
    msg = normalize_user_text(message).strip()
    contextual_references = {
        "שם", "בו", "בה", "שלו", "שלה", "עליו", "עליה",
        "סרט", "לסרט", "של סרט", "הסרט", "הסרט הזה", "בסרט", "בסרט הזה", "אותו", "אותה",
        "there", "it", "the movie", "this movie", "the film", "this film",
    }
    patterns = [
        r"מי משחק(?:ים)?\s+(?:בסרט\s+|ב)?(.+)",
        r"מי השחקנים\s+(?:בסרט\s+|ב)?(.+)",
        r"(?:מי ה)?שחקן הראשי\s+(?:בסרט\s+|ב)?(.+)",
        r"(?:מי ה)?שחקנית הראשית\s+(?:בסרט\s+|ב)?(.+)",
        r"מי ביים\s+(?:את\s+)?(.+)",
        r"על מה הסרט\s+(.+)",
        r"תקציר של\s+(.+)",
        r"(?:אתה יכול|את יכולה|אפשר)?\s*(?:לפרט|תפרט|תפרטי)\s+(?:לי\s+)?על\s+(.+)",
        r"(?:ספר|ספרי)\s+לי\s+על\s+(.+)",
        r"דירוג של\s+(.+)",
        r"האורך של\s+(.+)",
        r"סרטי המשך של\s+(.+)", r"סרט המשך של\s+(.+)", r"המשך של\s+(.+)",
        r"sequels? (?:of|to)\s+(.+)",
        r"similar to\s+(.+)", r"movies like\s+(.+)", r"movie like\s+(.+)",
        r"דומה ל\s+(.+)", r"דומים ל\s+(.+)", r"כמו\s+(.+)", r"בסגנון\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip("?.!.,")
            if normalize_user_text(candidate).lower() in contextual_references:
                return None
            return candidate

    low = msg.lower()
    for title in all_titles_no_year:
        if len(title) >= 4 and title in low:
            return title
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

    if len(title_no_year) >= 4:
        partial = df[df["title_no_year"].str.contains(re.escape(title_no_year), na=False)].index
        if len(partial) > 0:
            return int(partial[0])

    close = difflib.get_close_matches(title_no_year, all_titles_no_year, n=1, cutoff=0.72)
    if close:
        idx = df[df["title_no_year"] == close[0]].index
        if len(idx) > 0:
            return int(idx[0])
    return None

# ----------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------

def get_valid_gemini_api_key() -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if api_key in {"", "YOUR_KEY_HERE", "your_real_key_here", "המפתח_האמיתי_שלך", "המפתח האמיתי שלך"}:
        return None
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        return None
    if " " in api_key or len(api_key) < 20:
        return None
    return api_key


def model_fallback_list() -> List[str]:
    configured = os.getenv("GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL
    env_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
    models = [configured]
    if env_fallbacks:
        models += [m.strip() for m in env_fallbacks.split(",") if m.strip()]
    else:
        models += ["gemini-2.5-flash-lite", "gemini-2.0-flash"]
    unique = []
    for model in models:
        if model and model not in unique:
            unique.append(model)
    return unique


def should_try_next_gemini_model(error_text: str) -> bool:
    text = str(error_text).lower()
    return any(signal in text for signal in ["429", "resource_exhausted", "quota", "rate limit", "unavailable", "503", "not_found", "404", "not supported"])


def generate_with_gemini(contents: str, use_google_search: bool = False) -> Optional[str]:
    api_key = get_valid_gemini_api_key()
    if not api_key or genai is None:
        return None
    client = genai.Client(api_key=api_key)
    last_error = None
    for model_name in model_fallback_list():
        try:
            kwargs = {"model": model_name, "contents": contents}

            # For cinema requests we need fresh now-playing information.
            # The official Google GenAI SDK way is to pass a GoogleSearch tool
            # through GenerateContentConfig. This makes Gemini ground the answer
            # with current web results instead of relying on old model knowledge.
            if use_google_search:
                if types is not None:
                    grounding_tool = types.Tool(google_search=types.GoogleSearch())
                    kwargs["config"] = types.GenerateContentConfig(
                        tools=[grounding_tool],
                        temperature=0.2,
                    )
                else:
                    kwargs["config"] = {
                        "tools": [{"google_search": {}}],
                        "temperature": 0.2,
                    }

            response = client.models.generate_content(**kwargs)
            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip()
            last_error = f"empty response from {model_name}"
        except Exception as exc:
            last_error = str(exc)
            print(f"Gemini failed with {model_name}: {last_error}")
            if should_try_next_gemini_model(last_error):
                continue
            break
    print(f"Gemini unavailable. Last error: {last_error}")
    return None


def normalize_intent_with_gemini(user_text: str) -> str:
    """Correct free-form spelling for intent parsing without answering the user."""
    original = str(user_text or "").strip()
    if not original or not get_valid_gemini_api_key() or genai is None:
        return original
    if original in INTENT_NORMALIZATION_CACHE:
        return INTENT_NORMALIZATION_CACHE[original]

    prompt = f"""
Correct spelling and obvious typing mistakes in this Hebrew/English user message.
Preserve the exact meaning, intent, movie titles, actor names, numbers, and constraints.
Do not answer the message. Do not explain. Return only the corrected message.

Message:
{original}
""".strip()
    corrected = generate_with_gemini(prompt, use_google_search=False)
    if corrected:
        corrected = corrected.strip().strip('"').strip("'")
        if corrected and len(corrected) <= max(300, len(original) * 3):
            # Typo correction must never erase preferences that were already
            # explicit in the user's message.
            original_genres = set(detect_requested_genres(original))
            corrected_genres = set(detect_requested_genres(corrected))
            original_viewing = detect_viewing_preference(original)
            corrected_viewing = detect_viewing_preference(corrected)
            lost_genre = bool(original_genres - corrected_genres)
            lost_viewing = original_viewing in {"home", "cinema"} and corrected_viewing != original_viewing
            lost_runtime = extract_runtime_filter(original) != (0, 1000) and extract_runtime_filter(corrected) == (0, 1000)
            lost_age = extract_age_filter(original)[:2] != (None, None) and extract_age_filter(corrected)[:2] == (None, None)
            if lost_genre or lost_viewing or lost_runtime or lost_age:
                INTENT_NORMALIZATION_CACHE[original] = original
                return original
            INTENT_NORMALIZATION_CACHE[original] = corrected
            return corrected
    return original


def infer_viewing_preference_with_gemini(user_text: str) -> str:
    """Classify an unclear free-form answer by meaning, including typos."""
    text = str(user_text or "").strip()
    if not text or not get_valid_gemini_api_key() or genai is None:
        return "unclear"
    if text in VIEWING_INTENT_CACHE:
        cached = VIEWING_INTENT_CACHE[text]
        if cached in {"home", "cinema"}:
            return cached
        VIEWING_INTENT_CACHE.pop(text, None)

    prompt = f"""
Classify the user's intended movie viewing location despite spelling mistakes.
Infer the intended meaning semantically. A misspelled location word with an
extra, missing, swapped, or incorrect letter must still be classified by its
likely intended meaning. Do not require an exact dictionary spelling.
Return exactly one word:
HOME - the user wants to watch at home or streaming
CINEMA - the user wants to watch at a cinema/theater
UNCLEAR - neither is clearly intended

User message: {text}
""".strip()
    result = generate_with_gemini(prompt, use_google_search=False)
    normalized = str(result or "").strip().upper()
    if re.search(r"\bCINEMA\b", normalized):
        preference = "cinema"
    elif re.search(r"\bHOME\b", normalized):
        preference = "home"
    else:
        preference = "unclear"
    if preference != "unclear":
        VIEWING_INTENT_CACHE[text] = preference
    return preference


def build_constraints_text(user_message: str) -> str:
    genres = detect_requested_genres(user_message)
    actor = extract_requested_actor(user_message)
    min_runtime, max_runtime = extract_runtime_filter(user_message)
    min_rating = extract_min_rating(user_message)
    min_age, max_age, age_description = extract_age_filter(user_message)

    lines = []
    if genres:
        lines.append("ז׳אנר חובה: " + ", ".join(genres))
    if actor:
        lines.append("שחקן/שחקנית חובה: " + actor)
    if min_runtime > 0 or max_runtime < 1000:
        if min_runtime > 0 and max_runtime >= 1000:
            lines.append(f"אורך חובה: לפחות {min_runtime} דקות")
        elif max_runtime < 1000 and min_runtime <= 0:
            lines.append(f"אורך חובה: עד {max_runtime} דקות")
        else:
            lines.append(f"אורך חובה: בין {min_runtime} ל-{max_runtime} דקות")
    if min_rating is not None:
        lines.append(f"דירוג מינימלי: {min_rating}/10")
    if age_description:
        lines.append("מגבלת גיל: " + age_description)
    if min_age is not None:
        lines.append(f"גיל מינימלי: {min_age}+")
    if max_age is not None:
        lines.append(f"גיל מקסימלי: עד {max_age}")
    return "\n".join("- " + line for line in lines) if lines else "- אין פילטרים מובנים, ענה לפי הטקסט המלא."


def gemini_movie_fallback(user_message: str, task_name: str, cinema: bool = False) -> str:
    if cinema:
        search_instruction = """
The user chose CINEMA, so the local movie dataset is not relevant.
Use Google Search grounding to find movies that are actually playing in cinemas NOW, preferably in Israel because the conversation is in Hebrew.
The recommendations must match the user's mandatory constraints.
Do NOT recommend famous older movies unless the search results confirm they are currently screening.
Do NOT answer with a generic list of cinematic movies.
If you cannot verify enough matching now-playing movies, say in Hebrew that you could not verify enough currently-screening matches and ask the user to broaden the genre or city.
Return every verified recommendation in EXACTLY this format, with a blank line between movies:
🎬 English title (Hebrew title if known)
למה זה מתאים: concise Hebrew reason
קולנוע/אזור: cinema chain, branch or city
ז׳אנר: genre/style
אורך הסרט: runtime if verified
מקור מאומת: full direct URL to the current cinema or showtime page

Do not use markdown bullets or numbered lists. Do not include a movie without a direct verified source URL.
"""
    else:
        search_instruction = "Do not discuss platform availability. Just recommend movies that match the request."

    prompt = f"""
You are a friendly Hebrew movie recommendation agent.
Answer only about movies. Answer naturally in Hebrew.
Do not reveal whether you used a dataset, Gemini, Google Search, fallback, or internal logic.
Respect every constraint as mandatory. Do not fill the answer with unrelated movies.
If a requested actor is included, every recommendation must include that actor.
If a requested genre is included, every recommendation must match that genre.
If runtime/age constraints are included, keep them.
Give 3-5 concise recommendations.
{search_instruction}

User request:
{user_message}

Mandatory constraints:
{build_constraints_text(user_message)}

Task:
{task_name}
""".strip()

    answer = generate_with_gemini(prompt, use_google_search=cinema)
    if answer:
        cleaned = clean_user_answer(answer)
        # A cinema answer is useful only when it contains a verifiable current
        # showtime/cinema source. Reject generic model-knowledge movie lists.
        if not cinema or re.search(r"https?://", cleaned, flags=re.IGNORECASE):
            return cleaned
    if cinema:
        return ""
    return "לא מצאתי התאמה מספיק טובה כרגע. נסי לדייק לי ז׳אנר, אורך, גיל או שחקן/שחקנית מועדפים."


def clean_user_answer(text: str) -> str:
    if not text:
        return text
    forbidden = [
        "Gemini", "Google Search", "dataset", "CSV", "דאטה סט", "דאטהסט", "מאגר הנתונים",
        "fallback", "API", "local data", "נתונים מקומיים"
    ]
    cleaned = str(text)
    for word in forbidden:
        cleaned = cleaned.replace(word, "")
    return cleaned.strip()

# ----------------------------------------------------------------------
# Formatting and local CSV recommendations
# ----------------------------------------------------------------------

def format_movie(row: pd.Series) -> str:
    title = row.get("title_movielens", "Unknown Title")
    rating = float(row.get("vote_average", 0) or 0)
    runtime = float(row.get("runtime", 0) or 0)
    age_label = str(row.get("age_restriction_label", "")).strip()

    text = (
        f"🎬 {title}\n"
        f"אורך הסרט: {runtime:.0f} דקות\n"
        f"דירוג: {rating:.1f}/10"
    )
    if age_label:
        text += f"\nמגבלת גיל משוערת: {age_label}"
    return text


def apply_strict_filters(user_text: str) -> Tuple[np.ndarray, List[str]]:
    requested_genres = detect_requested_genres(user_text)
    requested_actor = extract_requested_actor(user_text)
    min_runtime, max_runtime = extract_runtime_filter(user_text)
    min_rating = extract_min_rating(user_text)
    min_age, max_age, age_description = extract_age_filter(user_text)

    mask = np.ones(len(df), dtype=bool)
    details = []

    if requested_genres:
        # Hard genre filter: every returned movie must include at least one requested genre.
        genre_mask = df["genres_list"].apply(
            lambda genres: any(g in set(str(x).strip() for x in genres) for g in requested_genres)
        ).values
        mask &= genre_mask
        details.append("ז׳אנר: " + ", ".join(requested_genres))

    if requested_actor:
        actor_mask = df["cast_names_lower"].fillna("").str.contains(re.escape(requested_actor.lower()), na=False).values
        mask &= actor_mask
        details.append("שחקן/שחקנית: " + requested_actor)

    if min_runtime > 0 or max_runtime < 1000:
        runtime_mask = ((df["runtime"] >= min_runtime) & (df["runtime"] <= max_runtime)).values
        mask &= runtime_mask
        if min_runtime > 0 and max_runtime >= 1000:
            details.append(f"אורך: מעל {min_runtime} דקות")
        elif max_runtime < 1000 and min_runtime <= 0:
            details.append(f"אורך: עד {max_runtime} דקות")
        else:
            details.append(f"אורך: {min_runtime}-{max_runtime} דקות")

    if min_rating is not None:
        mask &= (df["vote_average"] >= min_rating).values
        details.append(f"דירוג מינימלי: {min_rating}")

    if min_age is not None:
        mask &= (df["age_limit"] >= min_age).values
    if max_age is not None:
        mask &= (df["age_limit"] <= max_age).values
    if age_description:
        details.append("מגבלת גיל: " + age_description)

    return mask, details


def recommend_from_csv(user_text: str, n: int = 5) -> Tuple[str, str]:
    mask, details = apply_strict_filters(user_text)
    candidates = np.where(mask)[0]

    if len(candidates) == 0:
        return "", "data_missing"

    user_vector = vectorizer.transform([normalize_user_text(user_text)])
    text_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    rating_rank = df["vote_average"].rank(pct=True).values
    popularity_rank = df["popularity"].rank(pct=True).values
    vote_rank = df["vote_count"].rank(pct=True).values

    final_scores = 0.55 * text_scores + 0.20 * rating_rank + 0.15 * popularity_rank + 0.10 * vote_rank

    indices = candidates[np.argsort(final_scores[candidates])[::-1]][:n]
    results = [format_movie(df.iloc[idx]) for idx in indices]

    details_text = "\n" + " | ".join(details) if details else ""
    answer = (
        "מעולה, מצאתי כמה סרטים שיכולים להתאים למה שביקשת 😊"
        + details_text + "\n\n" + "\n\n".join(results)
        + "\n\nרוצה שאבחר לך אחד מהם ואספר עליו קצת יותר?"
    )
    return answer, "data_found"


def recommend_by_movie(movie_title: Optional[str], n: int = 5) -> Tuple[str, str]:
    movie_index = search_movie_index(movie_title)
    if movie_index is None:
        return "", "data_missing"

    text_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    numeric_scores = cosine_similarity(numeric_matrix[movie_index].reshape(1, -1), numeric_matrix).flatten()
    same_cluster_bonus = (df["cluster"].values == df.loc[movie_index, "cluster"]).astype(float) * 0.08
    final_scores = 0.72 * text_scores + 0.20 * numeric_scores + same_cluster_bonus

    indices = final_scores.argsort()[::-1]
    results = []
    for idx in indices:
        if idx == movie_index:
            continue
        results.append(format_movie(df.iloc[idx]))
        if len(results) == n:
            break

    selected = df.loc[movie_index, "title_movielens"]
    return f"מצאתי כמה סרטים שיכולים להתאים אם אהבת את {selected}:\n\n" + "\n\n".join(results), "data_found"


def find_confident_movie_sequels(movie_index: int) -> List[str]:
    """Return only titles that clearly identify themselves as film sequels."""
    row = df.iloc[movie_index]
    base_title = str(row.get("title_no_year", "")).strip()
    if not base_title:
        return []

    sequel_marker = r"(?:part\s+(?:[ivxlcdm]+|\d+)|(?:[ivxlcdm]+|\d+))"
    sequel_pattern = re.compile(
        rf"^{re.escape(base_title)}\s+{sequel_marker}(?:\b|:)",
        flags=re.IGNORECASE,
    )
    sequels = []
    for idx, candidate in df.iterrows():
        if idx == movie_index:
            continue
        candidate_title = str(candidate.get("title_no_year", "")).strip()
        if sequel_pattern.search(candidate_title):
            sequels.append(str(candidate.get("title_movielens", candidate_title)))
    return sequels[:8]


def get_movie_info(user_text: str, fallback_title: Optional[str] = None) -> Tuple[str, str]:
    msg = normalize_user_text(user_text).lower()
    title = find_title_in_message(user_text) or fallback_title
    idx = search_movie_index(title)
    if idx is None:
        return "", "data_missing"

    row = df.iloc[idx]
    movie_title = row.get("title_movielens", "Unknown Title")

    asks_director = any(term in msg for term in ["מי ביים", "במאי", "director", "who directed"])
    asks_lead_cast = any(term in msg for term in [
        "שחקן ראשי", "השחקן הראשי", "שחקנית ראשית", "השחקנית הראשית",
        "lead actor", "lead actress", "main actor",
    ])
    asks_cast = any(term in msg for term in ["מי משחק", "מי השחקנים", "שחקנים", "קאסט", "cast", "actor", "actors"])
    asks_plot = any(term in msg for term in [
        "על מה", "תקציר", "עלילה", "פירוט", "לפרט", "תפרט", "תפרטי",
        "ספר לי על", "ספרי לי על", "plot", "about",
    ])
    asks_sequels = any(term in msg for term in ["סרט המשך", "סרטי המשך", "המשך של", "sequel", "sequels"])
    asks_runtime = any(term in msg for term in ["אורך", "כמה זמן", "runtime", "length", "duration"])
    asks_rating = any(term in msg for term in ["דירוג", "rating", "review"])
    asks_age_suitability = any(term in msg for term in [
        "מתאים לילדים", "מתאים לילד", "מתאים לגיל", "מתחת לגיל", "מגבלת גיל",
        "suitable for children", "age rating",
    ])

    if asks_director and str(row.get("director_names", "")).strip():
        return f"{movie_title} בוים על ידי {row.get('director_names')}.", "data_found"
    if asks_lead_cast and str(row.get("cast_names", "")).strip():
        lead_cast = [name.strip() for name in str(row.get("cast_names", "")).split(",") if name.strip()][:3]
        if lead_cast:
            return f"השחקנים הראשיים ב-{movie_title} הם: {', '.join(lead_cast)}.", "data_found"
    if asks_cast and str(row.get("cast_names", "")).strip():
        return f"ב-{movie_title} משחקים: {row.get('cast_names')}.", "data_found"
    if asks_plot:
        overview = str(row.get("overview", "")).strip()
        if overview:
            return f"תקציר של {movie_title}:\n{overview}", "data_found"
    if asks_sequels:
        sequel_note = KNOWN_SEQUEL_NOTES.get(str(row.get("title_no_year", "")).strip().lower())
        if sequel_note:
            return sequel_note, "data_found"
        sequels = find_confident_movie_sequels(idx)
        if sequels:
            return (
                f"כן, קיימים סרטי המשך ל-{movie_title}:\n"
                + "\n".join(f"- {title}" for title in sequels)
            ), "data_found"
        # Similar words in a title do not prove a sequel relationship. Let
        # Gemini distinguish an official film sequel from a TV spin-off or
        # another unrelated title.
        return "", "data_missing"
    if asks_runtime:
        return f"האורך של {movie_title} הוא בערך {float(row.get('runtime', 0)):.0f} דקות.", "data_found"
    if asks_rating:
        return f"הדירוג של {movie_title} הוא בערך {float(row.get('vote_average', 0)):.1f}/10.", "data_found"
    if asks_age_suitability:
        age_label = str(row.get("age_restriction_label", "")).strip() or "לא ידועה"
        age_limit = int(row.get("age_limit", 13) or 13)
        if age_limit <= 0:
            return f"כן. {movie_title} מסומן כמתאים לכל המשפחה.", "data_found"
        return (
            f"{movie_title} מסומן לגילאי {age_label}. "
            f"לכן הוא מתאים מגיל {age_limit} ומעלה, אך לא לכל הילדים שמתחת לגיל 18."
        ), "data_found"

    overview = str(row.get("overview", "")).strip()
    if overview:
        return f"תקציר של {movie_title}:\n{overview}", "data_found"
    return f"לא מצאתי תקציר זמין ל-{movie_title}.", "data_found"


def show_trends() -> Tuple[str, str]:
    genres_series = df["genres_movielens"].str.split("|").explode()
    top_genres = genres_series.value_counts().head(10)
    top_movies = df.sort_values(by="popularity", ascending=False).head(5)

    response = "📊 מגמות ודפוסים בדאטה:\n\n"
    response += "הז׳אנרים הנפוצים ביותר:\n"
    response += "\n".join(f"- {genre}: {count} movies" for genre, count in top_genres.items())
    response += "\n\nהסרטים הפופולריים ביותר:\n"
    response += "\n".join(
        f"- {row['title_movielens']} | Popularity: {float(row['popularity']):.2f} | Rating: {float(row['vote_average']):.2f}"
        for _, row in top_movies.iterrows()
    )
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

# ----------------------------------------------------------------------
# Conversation flow
# ----------------------------------------------------------------------

def build_home_or_cinema_question(user_text: str) -> str:
    genres = detect_requested_genres(user_text)
    actor = extract_requested_actor(user_text)
    min_runtime, max_runtime = extract_runtime_filter(user_text)
    _, _, age_description = extract_age_filter(user_text)

    details = []
    if genres:
        details.append("ז׳אנר: " + ", ".join(genres))
    if actor:
        details.append("שחקן/שחקנית: " + actor)
    if min_runtime > 0 or max_runtime < 1000:
        if min_runtime > 0 and max_runtime >= 1000:
            details.append(f"אורך: מעל {min_runtime} דקות")
        elif max_runtime < 1000 and min_runtime <= 0:
            details.append(f"אורך: עד {max_runtime} דקות")
        else:
            details.append(f"אורך: {min_runtime}-{max_runtime} דקות")
    if age_description:
        details.append("מגבלת גיל: " + age_description)

    details_text = "\n" + " | ".join(details) if details else ""
    genre_question = "\nאיזה ז׳אנר או סגנון בא לך לראות?" if not genres else ""
    return (
        "מעולה 😊 כדי לתת לך המלצה רלוונטית, איפה תרצי/תרצה לראות את הסרט?"
        + details_text +
        "\n\nאפשר לבחור:\n"
        "• בבית — אני אחפש סרטים שמתאימים לצפייה ביתית.\n"
        "• בקולנוע — אני אבדוק סרטים שמתאימים למה שביקשת ומוקרנים עכשיו.\n\n"
        "אפשר לענות פשוט: \"בבית\" או \"בקולנוע\"."
    )


def answer_recommendation(user_text: str, session_id: str, viewing_mode: str) -> str:
    # Once a recommendation path starts, the home/cinema question has been
    # answered and must not leak into later follow-up requests.
    get_state(session_id).pop("pending_recommendation_request", None)
    get_state(session_id).pop("pending_viewing_mode", None)
    get_state(session_id).pop("pending_needs_genre", None)

    if viewing_mode == "cinema":
        if not has_concrete_movie_preferences(user_text):
            return ask_cinema_preferences(session_id, user_text)
        get_state(session_id).pop("waiting_for_cinema_preferences", None)
        save_last_context(session_id, user_text, "cinema")
        cinema_answer = gemini_movie_fallback(user_text, "current cinema movie recommendation", cinema=True)
        if cinema_answer:
            get_state(session_id).pop("last_cinema_search_failed", None)
            return cinema_answer
        get_state(session_id)["last_cinema_search_failed"] = "1"
        return (
            "שמרתי את ההעדפות שלך לקולנוע, אבל כרגע לא הצלחתי להתחבר לחיפוש האינטרנט "
            "כדי לבדוק אילו סרטים באמת מוקרנים עכשיו. נסי שוב בעוד רגע."
        )

    # Home viewing searches the local dataset first, then asks Gemini only
    # when no matching local movies exist.
    local_answer, status = recommend_from_csv(user_text)
    save_last_context(session_id, user_text, "home")
    if status == "data_found":
        return local_answer
    return gemini_movie_fallback(user_text, "movie recommendation", cinema=False)


def answer_movie_detail(user_text: str, session_id: str) -> str:
    explicit_title = find_title_in_message(user_text)
    focused_title = explicit_title or get_focused_movie(session_id)
    local_answer, status = get_movie_info(user_text, fallback_title=focused_title)
    if status == "data_found":
        if focused_title:
            save_focused_movie(session_id, focused_title)
        return local_answer

    if not focused_title:
        get_state(session_id)["pending_movie_detail_question"] = user_text
        return "על איזה סרט מדובר? כתבי לי את שם הסרט ואמשיך משם."

    context_line = f"The movie being discussed is: {focused_title}.\n" if focused_title else ""
    concise_request = (
        f"{context_line}Answer ONLY the user's current question, concisely. "
        f"Do not repeat prior details, plot, cast, or sequels unless explicitly requested. "
        f"For sequel questions, list only official feature-film sequels. Clearly distinguish "
        f"TV series, spin-offs, and unrelated titles that merely share words in the title.\n"
        f"Current question: {user_text}"
    )
    return gemini_movie_fallback(concise_request, "concise movie information", cinema=False)


def unrelated_response() -> str:
    return (
        "זה לא התחום שאני מתעסק בו — אני סוכן שמתמחה בהמלצות ומידע על סרטים 🎬\n"
        "אפשר לבקש ממני למשל סרט קומדיה, סרט עם שחקן אהוב, או סרט שמתאים לכל המשפחה."
    )


def is_greeting(user_text: str) -> bool:
    text = re.sub(r"[^\wא-ת]+", " ", str(user_text or "").lower()).strip()
    greetings = {
        "היי", "הי", "שלום", "אהלן", "מה קורה", "מה נשמע", "בוקר טוב",
        "צהריים טובים", "ערב טוב", "לילה טוב", "hello", "hi", "hey",
    }
    return text in greetings


def greeting_response() -> str:
    return (
        "היי, מה קורה? 😊\n"
        "במה אוכל לעזור לך? אני סרטלי, ואני יודע לעזור בהמלצות על סרטים, "
        "למצוא סרט לפי מצב הרוח שלך ולענות על שאלות מעולם הקולנוע 🎬"
    )


def is_thanks(user_text: str) -> bool:
    text = re.sub(r"[^\wא-ת]+", " ", str(user_text or "").lower()).strip()
    thanks_terms = ["תודה", "תודה רבה", "תודה גבר", "תודה אחי", "תודה מלך", "thanks", "thank you", "thx"]
    return any(text == term or text.startswith(term + " ") for term in thanks_terms)


def thanks_response() -> str:
    return "בכיף, בשמחה! תמיד פה לכל מה שצריך 😊🎬"


def movie_agent(user_message: str, session_id: str = "local") -> str:
    raw_message = str(user_message or "").strip()
    original_message = normalize_intent_with_gemini(raw_message)
    if not original_message:
        return "כתבי לי איזה סרט בא לך לראות 🎬"

    msg = normalize_user_text(original_message).lower()
    current_preference = detect_viewing_preference(original_message)

    pending_detail_question = get_state(session_id).get("pending_movie_detail_question")
    supplied_title = raw_message.strip().strip("?.!.,")
    if pending_detail_question:
        get_state(session_id).pop("pending_movie_detail_question", None)
        save_focused_movie(session_id, supplied_title)
        local_answer, status = get_movie_info(pending_detail_question, fallback_title=supplied_title)
        if status == "data_found":
            return local_answer
        return answer_movie_detail(pending_detail_question, session_id)

    # Preserve an explicitly written movie title exactly as the user entered
    # it. Gemini typo correction can otherwise alter uncommon titles before
    # the local dataset lookup.
    raw_explicit_title = find_title_in_message(raw_message)
    if raw_explicit_title and is_detail_question(raw_message):
        save_focused_movie(session_id, raw_explicit_title)
        return answer_movie_detail(raw_message, session_id)

    last_viewing_mode = get_last_viewing_mode(session_id)
    if last_viewing_mode in {"home", "cinema"} and is_collection_recommendation_filter(raw_message):
        combined_request = merge_with_movie_context(session_id, original_message)
        return answer_recommendation(combined_request, session_id, last_viewing_mode)

    # A detail question has priority unless the user explicitly asks for
    # recommendations that use the detail as a collection filter, such as
    # "recommend movies rated above 9".
    if is_detail_question(original_message) and not has_recommendation_intent(original_message):
        return answer_movie_detail(original_message, session_id)

    if is_greeting(original_message):
        return greeting_response()

    if is_thanks(original_message):
        return thanks_response()

    # Generic recommendation words must not turn an unrelated request into a movie request.
    if is_explicitly_non_movie_related(original_message):
        get_state(session_id).pop("pending_recommendation_request", None)
        return unrelated_response()

    # A direct home/cinema answer is always a viewing choice, never a movie
    # detail question. This also recovers if the development server restarted
    # and lost an in-memory pending recommendation.
    if current_preference in {"home", "cinema"} and not get_state(session_id).get("pending_recommendation_request"):
        movie_context = get_last_context(session_id)
        if movie_context and has_genre_or_style_preferences(movie_context):
            return answer_recommendation(movie_context, session_id, current_preference)
        return ask_genre_question(session_id, movie_context or "המלצה על סרט", current_preference)

    # We already know the viewing location and are waiting only for a genre/style.
    pending_viewing_mode = get_state(session_id).get("pending_viewing_mode")
    if pending_viewing_mode in {"home", "cinema"}:
        pending = pop_pending_request(session_id) or ""
        combined_request = f"{pending}. ז׳אנר או סגנון נוסף: {original_message}".strip(". ")
        if not has_genre_or_style_preferences(combined_request):
            return ask_genre_question(session_id, combined_request, pending_viewing_mode)
        get_state(session_id).pop("pending_viewing_mode", None)
        get_state(session_id).pop("pending_needs_genre", None)
        return answer_recommendation(combined_request, session_id, pending_viewing_mode)

    if get_state(session_id).pop("waiting_for_cinema_preferences", None):
        combined_request = merge_with_movie_context(session_id, original_message)
        if has_concrete_movie_preferences(combined_request):
            return answer_recommendation(combined_request, session_id, "cinema")
        return ask_cinema_preferences(session_id, combined_request)

    # A viewing-mode change in the middle of the conversation keeps all
    # collected movie preferences and immediately searches using the new mode.
    if (
        current_preference in {"home", "cinema"}
        and get_last_context(session_id)
        and not get_state(session_id).get("pending_recommendation_request")
    ):
        pending_context = get_state(session_id).pop("pending_recommendation_request", None)
        movie_context = pending_context or get_last_context(session_id)
        return answer_recommendation(movie_context, session_id, current_preference)

    # 1. If we are waiting for home/cinema, combine the new answer with the original request.
    pending = pop_pending_request(session_id)
    if pending:
        pending_needs_genre = get_state(session_id).pop("pending_needs_genre", "0") == "1"
        if current_preference == "unclear":
            current_preference = infer_viewing_preference_with_gemini(original_message)
        if current_preference in {"home", "cinema"}:
            combined_request = f"{pending}. בחירת צפייה: {current_preference}."
            if pending_needs_genre:
                return ask_genre_question(session_id, combined_request, current_preference)
            return answer_recommendation(combined_request, session_id, current_preference)
        # User gave more filters instead of choosing. Keep everything and ask again.
        combined_request = f"{pending}. דרישה נוספת: {original_message}"
        save_pending_request(session_id, combined_request)
        return build_home_or_cinema_question(combined_request)

    # Detail questions about the focused movie take priority over recommendation
    # filters, especially questions containing words such as "שחקן".
    if is_detail_question(original_message) and (
        not has_recommendation_intent(original_message)
        or (get_focused_movie(session_id) and refers_to_focused_movie(original_message))
    ):
        return answer_movie_detail(original_message, session_id)

    # Only clear refinement constraints continue the active recommendation.
    # A new genre request starts a new recommendation and asks home/cinema.
    last_viewing_mode = get_last_viewing_mode(session_id)
    if last_viewing_mode in {"home", "cinema"} and is_genre_or_style_only(str(user_message or "")):
        combined_request = merge_with_movie_context(session_id, original_message)
        return answer_recommendation(combined_request, session_id, last_viewing_mode)

    if (
        last_viewing_mode == "cinema"
        and get_state(session_id).pop("last_cinema_search_failed", None)
        and has_genre_or_style_preferences(original_message)
    ):
        combined_request = merge_with_movie_context(session_id, original_message)
        return answer_recommendation(combined_request, session_id, "cinema")

    if last_viewing_mode in {"home", "cinema"} and is_recommendation_followup_filter(original_message):
        combined_request = merge_with_movie_context(session_id, original_message)
        return answer_recommendation(combined_request, session_id, last_viewing_mode)

    # 3. Similarity / analysis tasks.
    if any(term in msg for term in ["cluster", "clusters", "clustering", "קלאסטר", "אשכול"]):
        local, _ = describe_clusters()
        return local

    if any(term in msg for term in ["trend", "trends", "popular", "pattern", "מגמות", "דפוסים", "פופולרי"]):
        local, _ = show_trends()
        return local

    if any(term in msg for term in ["similar to", "movies like", "movie like", "דומה ל", "דומים ל", "כמו", "בסגנון"]):
        local, status = recommend_by_movie(find_title_in_message(original_message))
        if status == "data_found":
            save_last_context(session_id, original_message)
            return local
        return gemini_movie_fallback(original_message, "similar movie recommendation", cinema=False)

    # 4. Recommendations: keep the request, ask home/cinema if not already specified.
    if has_recommendation_intent(original_message):
        state = get_state(session_id)
        state.pop("last_movie_context", None)
        state.pop("last_viewing_mode", None)
        state.pop("pending_recommendation_request", None)
        state.pop("focused_movie_title", None)
        if current_preference in {"home", "cinema"}:
            if not has_genre_or_style_preferences(original_message):
                return ask_genre_question(session_id, original_message, current_preference)
            return answer_recommendation(original_message, session_id, current_preference)
        save_pending_request(session_id, original_message)
        return build_home_or_cinema_question(original_message)

    # 5. Other movie-related questions: CSV first, then Gemini.
    if is_movie_related(original_message):
        return answer_movie_detail(original_message, session_id)

    return unrelated_response()

# ----------------------------------------------------------------------
# Flask routes
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
        print(f"Chat route error: {exc}")
        reply = "נתקלתי בתקלה רגעית בעיבוד הבקשה. נסי לנסח שוב, למשל: ‘סרט קומדיה בבית’ או ‘סרט אימה בקולנוע’."
    return jsonify({"reply": reply})


@app.route("/reset", methods=["POST"])
def reset_chat():
    session_id = request.headers.get("X-Forwarded-For", request.remote_addr or "local")
    CONVERSATION_STATE.pop(session_id, None)
    return jsonify({"status": "ok"})


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
    app.run(host="127.0.0.1", port=5000, debug=True)

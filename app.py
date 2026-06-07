import re
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# Movie Recommendation Agent
# ---------------------------------------------------
# Required files in the same folder:
# - movies_agent_clean_data.csv
# - templates/index.html
# - static/style.css
# ---------------------------------------------------

app = Flask(__name__)

DATA_PATH = "movies_agent_clean_data.csv"

# ---------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["combined_text"] = df["combined_text"].fillna("")
df["genres_movielens"] = df["genres_movielens"].fillna("Unknown")
df["title_movielens"] = df["title_movielens"].fillna("Unknown Title")

numeric_features = ["runtime", "popularity", "vote_average", "vote_count"]

for col in numeric_features:
    if col not in df.columns:
        df[col] = 0

    df[col] = pd.to_numeric(df[col], errors="coerce")
    median_value = df[col].median()

    if pd.isna(median_value):
        median_value = 0

    df[col] = df[col].fillna(median_value)

# ---------------------------------------------------
# 2. Reduce dataset size for faster agent response
# ---------------------------------------------------
# The original dataset contains more than 27,000 movies.
# For a responsive web agent, we use the top 5,000 movies
# according to vote_count and popularity.

df = df.sort_values(
    by=["vote_count", "popularity"],
    ascending=False
).head(5000).reset_index(drop=True)

print("Dataset reduced for faster agent response.")
print("Number of movies used by agent:", len(df))

# ---------------------------------------------------
# 3. Build AI/NLP models
# ---------------------------------------------------

tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(df[numeric_features])

print("TF-IDF and numeric models are ready.")

# ---------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------

def format_movie_row(row, score=None):
    title = row.get("title_movielens", "Unknown Title")
    genres = row.get("genres_movielens", "Unknown")
    rating = row.get("vote_average", 0)
    popularity = row.get("popularity", 0)

    if score is not None:
        return (
            f"🎬 {title}\n"
            f"   Genres: {genres}\n"
            f"   Rating: {round(float(rating), 2)} | Popularity: {round(float(popularity), 2)}\n"
            f"   Match score: {round(float(score), 3)}"
        )

    return (
        f"🎬 {title}\n"
        f"   Genres: {genres}\n"
        f"   Rating: {round(float(rating), 2)} | Popularity: {round(float(popularity), 2)}"
    )


def find_movie_title_in_message(message):
    msg = message.strip()

    patterns = [
        r"similar to\s+(.+)",
        r"movies like\s+(.+)",
        r"movie like\s+(.+)",
        r"like\s+(.+)",
        r"דומים ל\s+(.+)",
        r"דומה ל\s+(.+)",
        r"כמו\s+(.+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def search_movie_index(movie_title):
    if not movie_title:
        return None

    title_lower = movie_title.lower().strip()

    exact_match = df[df["title_movielens"].str.lower() == title_lower].index

    if len(exact_match) > 0:
        return exact_match[0]

    partial_match = df[
        df["title_movielens"].str.lower().str.contains(
            re.escape(title_lower),
            na=False
        )
    ].index

    if len(partial_match) > 0:
        return partial_match[0]

    return None


# ---------------------------------------------------
# 5. Recommendation by movie title
# ---------------------------------------------------

def recommend_by_movie(movie_title, num_recommendations=5):
    movie_index = search_movie_index(movie_title)

    if movie_index is None:
        return (
            "לא מצאתי את הסרט הזה בדאטה.\n"
            "נסי לכתוב את השם באנגלית או להשתמש בשם מלא, למשל:\n"
            "similar to Toy Story"
        )

    text_similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    numeric_similarity_scores = cosine_similarity(
        numeric_matrix[movie_index].reshape(1, -1),
        numeric_matrix
    ).flatten()

    final_scores = 0.75 * text_similarity_scores + 0.25 * numeric_similarity_scores

    top_indices = final_scores.argsort()[::-1]
    results = []

    for index in top_indices:
        if index == movie_index:
            continue

        results.append(format_movie_row(df.iloc[index], final_scores[index]))

        if len(results) == num_recommendations:
            break

    selected_title = df.iloc[movie_index]["title_movielens"]

    return (
        f"הבנתי שאת מחפשת סרטים דומים ל: {selected_title}\n\n"
        "ההמלצות מבוססות על דמיון בתוכן הסרט, ז׳אנרים, תקציר, מילות מפתח "
        "וגם נתונים מספריים כמו דירוג ופופולריות:\n\n"
        + "\n\n".join(results)
    )


# ---------------------------------------------------
# 6. Recommendation by free text
# ---------------------------------------------------

def recommend_by_text(user_text, num_recommendations=5):
    user_vector = tfidf.transform([user_text])
    text_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    normalized_popularity = df["popularity"].rank(pct=True).values
    normalized_rating = df["vote_average"].rank(pct=True).values

    final_scores = (
        0.8 * text_scores +
        0.1 * normalized_popularity +
        0.1 * normalized_rating
    )

    top_indices = final_scores.argsort()[::-1][:num_recommendations]

    results = []

    for index in top_indices:
        results.append(format_movie_row(df.iloc[index], final_scores[index]))

    return (
        "הבנתי את הבקשה שלך מתוך הטקסט החופשי.\n"
        "הנה סרטים שיכולים להתאים לפי מילים, נושאים, ז׳אנרים ודירוגים:\n\n"
        + "\n\n".join(results)
    )


# ---------------------------------------------------
# 7. Trends and patterns
# ---------------------------------------------------

def show_trends():
    genres_series = df["genres_movielens"].str.split("|").explode()
    top_genres = genres_series.value_counts().head(10)

    top_movies = df[
        ["title_movielens", "popularity", "vote_average", "vote_count"]
    ].sort_values(by="popularity", ascending=False).head(5)

    high_rated = df[df["vote_count"] > 1000][
        ["title_movielens", "vote_average", "vote_count", "popularity"]
    ].sort_values(by="vote_average", ascending=False).head(5)

    response = "📊 זיהוי דפוסים ומגמות בדאטה:\n\n"

    response += "הז׳אנרים הנפוצים ביותר:\n"
    for genre, count in top_genres.items():
        response += f"- {genre}: {count} movies\n"

    response += "\nהסרטים הפופולריים ביותר:\n"
    for _, row in top_movies.iterrows():
        response += (
            f"- {row['title_movielens']} | "
            f"Popularity: {round(float(row['popularity']), 2)} | "
            f"Rating: {round(float(row['vote_average']), 2)}\n"
        )

    response += "\nסרטים עם דירוג גבוה ומספר הצבעות משמעותי:\n"
    for _, row in high_rated.iterrows():
        response += (
            f"- {row['title_movielens']} | "
            f"Rating: {round(float(row['vote_average']), 2)} | "
            f"Votes: {int(row['vote_count'])}\n"
        )

    return response


# ---------------------------------------------------
# 8. Anomaly detection
# ---------------------------------------------------

def detect_anomalies():
    popular_low_rating = df[
        (df["popularity"] > df["popularity"].quantile(0.90)) &
        (df["vote_average"] < 5)
    ][["title_movielens", "popularity", "vote_average", "vote_count"]].head(5)

    high_rating_few_votes = df[
        (df["vote_average"] >= 8) &
        (df["vote_count"] < 50)
    ][["title_movielens", "vote_average", "vote_count", "popularity"]].head(5)

    long_movies = df[
        df["runtime"] > df["runtime"].quantile(0.95)
    ][["title_movielens", "runtime", "vote_average", "popularity"]].sort_values(
        by="runtime",
        ascending=False
    ).head(5)

    response = "🔎 זיהוי אנומליות בדאטה:\n\n"

    response += "1. סרטים פופולריים עם דירוג נמוך:\n"
    if popular_low_rating.empty:
        response += "- לא נמצאו חריגות משמעותיות בקטגוריה זו.\n"
    else:
        for _, row in popular_low_rating.iterrows():
            response += (
                f"- {row['title_movielens']} | "
                f"Popularity: {round(float(row['popularity']), 2)} | "
                f"Rating: {round(float(row['vote_average']), 2)}\n"
            )

    response += "\n2. סרטים עם דירוג גבוה אך מעט הצבעות:\n"
    if high_rating_few_votes.empty:
        response += "- לא נמצאו חריגות משמעותיות בקטגוריה זו.\n"
    else:
        for _, row in high_rating_few_votes.iterrows():
            response += (
                f"- {row['title_movielens']} | "
                f"Rating: {round(float(row['vote_average']), 2)} | "
                f"Votes: {int(row['vote_count'])}\n"
            )

    response += "\n3. סרטים ארוכים במיוחד:\n"
    if long_movies.empty:
        response += "- לא נמצאו חריגות משמעותיות בקטגוריה זו.\n"
    else:
        for _, row in long_movies.iterrows():
            response += (
                f"- {row['title_movielens']} | "
                f"Runtime: {round(float(row['runtime']), 2)} minutes\n"
            )

    return response


# ---------------------------------------------------
# 9. Agent reasoning logic
# ---------------------------------------------------

def movie_agent(user_message):
    msg = user_message.lower().strip()

    if msg in ["hi", "hello", "hey", "שלום", "היי", "הי"]:
        return (
            "היי! אני סוכן המלצות סרטים 🎬\n"
            "אפשר לכתוב לי בצורה חופשית מה בא לך לראות, למשל:\n"
            "- בא לי קומדיה קלילה לערב\n"
            "- recommend a sci-fi movie about space\n"
            "- similar to Toy Story\n"
            "- show trends\n"
            "- detect anomalies"
        )

    if (
        "similar to" in msg or
        "movies like" in msg or
        "movie like" in msg or
        "דומה ל" in msg or
        "דומים ל" in msg or
        "כמו" in msg
    ):
        title = find_movie_title_in_message(user_message)
        return recommend_by_movie(title)

    if (
        "trend" in msg or
        "trends" in msg or
        "popular" in msg or
        "pattern" in msg or
        "מגמות" in msg or
        "דפוסים" in msg or
        "פופולרי" in msg
    ):
        return show_trends()

    if (
        "anomaly" in msg or
        "anomalies" in msg or
        "outlier" in msg or
        "outliers" in msg or
        "חריג" in msg or
        "חריגות" in msg or
        "אנומל" in msg
    ):
        return detect_anomalies()

    if (
        "recommend" in msg or
        "recommendation" in msg or
        "movie" in msg or
        "film" in msg or
        "בא לי" in msg or
        "רוצה" in msg or
        "תמליץ" in msg or
        "סרט" in msg
    ):
        return recommend_by_text(user_message)

    return (
        "אני כאן כדי לעזור לך למצוא סרטים ולהפיק תובנות מהדאטה 🎬\n"
        "אפשר לכתוב לי חופשי, למשל:\n"
        "- בא לי סרט רומנטי ומצחיק\n"
        "- recommend an action movie with superheroes\n"
        "- similar to Avatar\n"
        "- show trends\n"
        "- detect anomalies"
    )


# ---------------------------------------------------
# 10. Flask routes
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"reply": "כתבי לי מה בא לך לראות ואנסה להמליץ 😊"})

    reply = movie_agent(user_message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

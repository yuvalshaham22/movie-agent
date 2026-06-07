"""
Data preparation for Movie Recommendation AI Agent.
Combines several sources into one clean file:
- MovieLens movies: movie.csv
- MovieLens links: link.csv
- User tags: tag.csv
- TMDB movies: tmdb_5000_movies.csv
- TMDB credits: tmdb_5000_credits.csv
- Optional genome tags/scores: genome_tags.csv, genome_scores.csv

Output:
- movies_agent_clean_data.csv
"""

import ast
import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def read_csv_if_exists(filename: str) -> pd.DataFrame:
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {filename}")
    return pd.read_csv(path)


def extract_names(value):
    try:
        if pd.isna(value):
            return []
        items = ast.literal_eval(value)
        return [item.get("name", "") for item in items if isinstance(item, dict)]
    except Exception:
        return []


def main():
    print("Loading raw datasets...")
    movies_ml = read_csv_if_exists("movie.csv")
    links = read_csv_if_exists("link.csv")
    tags = read_csv_if_exists("tag.csv")
    tmdb_movies = read_csv_if_exists("tmdb_5000_movies.csv")
    tmdb_credits = read_csv_if_exists("tmdb_5000_credits.csv")

    # Optional genome files. The project still runs without them.
    genome_tags_path = BASE_DIR / "genome_tags.csv"
    genome_scores_path = BASE_DIR / "genome_scores.csv"
    genome_text = pd.DataFrame(columns=["movieId", "genome_tags_text"])

    if genome_tags_path.exists() and genome_scores_path.exists():
        print("Loading optional genome data...")
        genome_tags = pd.read_csv(genome_tags_path)
        genome_scores = pd.read_csv(genome_scores_path)
        top_scores = genome_scores.sort_values("relevance", ascending=False).groupby("movieId").head(8)
        top_scores = top_scores.merge(genome_tags, on="tagId", how="left")
        genome_text = top_scores.groupby("movieId")["tag"].apply(lambda s: " ".join(s.dropna().astype(str))).reset_index()
        genome_text = genome_text.rename(columns={"tag": "genome_tags_text"})

    print("Cleaning data...")
    movies_ml = movies_ml.drop_duplicates()
    links = links.drop_duplicates()
    tags = tags.drop_duplicates()
    tmdb_movies = tmdb_movies.drop_duplicates()
    tmdb_credits = tmdb_credits.drop_duplicates()

    movies_ml["genres"] = movies_ml["genres"].replace("(no genres listed)", "Unknown").fillna("Unknown")
    tags["tag"] = tags["tag"].fillna("")

    tmdb_movies["overview"] = tmdb_movies["overview"].fillna("")
    tmdb_movies["tagline"] = tmdb_movies["tagline"].fillna("")
    tmdb_movies["runtime"] = pd.to_numeric(tmdb_movies["runtime"], errors="coerce")
    tmdb_movies["runtime"] = tmdb_movies["runtime"].fillna(tmdb_movies["runtime"].median())
    tmdb_movies["release_date"] = tmdb_movies["release_date"].fillna("Unknown")

    # Aggregate user tags to movie level.
    user_tags = tags.groupby("movieId")["tag"].apply(lambda s: " ".join(s.dropna().astype(str).unique()[:40])).reset_index()
    user_tags = user_tags.rename(columns={"tag": "user_tags_text"})

    print("Merging sources...")
    links = links.rename(columns={"tmdbId": "id"})
    tmdb_credits = tmdb_credits.rename(columns={"movie_id": "id"})

    df = movies_ml.merge(links, on="movieId", how="left")
    df = df.merge(tmdb_movies, on="id", how="left", suffixes=("_movielens", "_tmdb"))
    df = df.merge(tmdb_credits, on="id", how="left")
    df = df.merge(user_tags, on="movieId", how="left")
    df = df.merge(genome_text, on="movieId", how="left")

    df["tmdb_genres_clean"] = df.get("genres_tmdb", "").apply(extract_names)
    df["keywords_clean"] = df.get("keywords", "").apply(extract_names)
    df["user_tags_text"] = df["user_tags_text"].fillna("")
    df["genome_tags_text"] = df["genome_tags_text"].fillna("")

    df["combined_text"] = (
        df["title_movielens"].fillna("").astype(str) + " " +
        df["genres_movielens"].fillna("").astype(str) + " " +
        df["overview"].fillna("").astype(str) + " " +
        df["tagline"].fillna("").astype(str) + " " +
        df["keywords_clean"].apply(lambda x: " ".join(x)) + " " +
        df["user_tags_text"].astype(str) + " " +
        df["genome_tags_text"].astype(str)
    )

    df = df.dropna(subset=["title_movielens"])
    df = df.drop_duplicates(subset=["title_movielens"])

    output_path = BASE_DIR / "movies_agent_clean_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved clean data to {output_path}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


if __name__ == "__main__":
    main()

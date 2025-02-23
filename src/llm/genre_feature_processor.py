from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd


class OptimizedGenreProcessor:
    """Efficient genre feature processing with caching."""

    def __init__(self):
        self.genre_cache = {}  # Cache for genre statistics
        self.similarity_cache = {}  # Cache for genre similarities
        self.cache_key = None  # Key for validating cache

    def _get_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key from DataFrame state."""
        return f"{len(df)}_{df['timestamp'].max()}_{df['rating'].mean()}"

    def _vectorize_genre_stats(self, user_history: pd.DataFrame) -> Dict:
        """Calculate genre statistics using vectorized operations."""
        # Create genre-rating pairs
        genre_ratings = []

        for _, row in user_history.iterrows():
            if isinstance(row["genres"], list):
                for genre in row["genres"]:
                    genre_ratings.append((genre, row["rating"], row["timestamp"]))

        if not genre_ratings:
            return {}

        # Convert to DataFrame for vectorized operations
        genre_df = pd.DataFrame(genre_ratings, columns=["genre", "rating", "timestamp"])

        # Group by genre and calculate stats
        stats = (
            genre_df.groupby("genre")
            .agg({"rating": ["count", "mean", "std"], "timestamp": "max"})
            .fillna(0)
        )

        # Calculate time weights once
        max_time = genre_df["timestamp"].max()
        genre_df["time_weight"] = np.exp(
            -0.1 * (max_time - genre_df["timestamp"]) / (24 * 60 * 60)
        )

        # Calculate weighted stats
        weighted_stats = genre_df.groupby("genre").agg(
            {
                "rating": lambda x: np.average(
                    x, weights=genre_df.loc[x.index, "time_weight"]
                ),
                "time_weight": "sum",
            }
        )

        # Combine stats
        result = {}
        for genre in stats.index:
            count = stats.loc[genre, ("rating", "count")]
            result[genre] = {
                "count": int(count),
                "avg_rating": float(stats.loc[genre, ("rating", "mean")]),
                "rating_std": float(stats.loc[genre, ("rating", "std")]),
                "weighted_avg": float(weighted_stats.loc[genre, "rating"]),
                "confidence": min(1.0, count / 10),
            }

        return result

    def _fast_similarity_calculation(
        self, user_history: pd.DataFrame
    ) -> Tuple[Dict, Set]:
        """Calculate genre similarities efficiently."""
        # Get all genres
        all_genres = set()
        movie_genres = defaultdict(list)
        movie_ratings = {}

        # Single pass through data
        for _, row in user_history.iterrows():
            if isinstance(row["genres"], list):
                movie_id = row.name
                genres = row["genres"]
                rating = row["rating"]

                movie_genres[movie_id] = genres
                movie_ratings[movie_id] = rating
                all_genres.update(genres)

        # Calculate co-occurrences
        cooccurrence = defaultdict(lambda: defaultdict(float))
        for movie_id, genres in movie_genres.items():
            if len(genres) < 2:
                continue

            rating_weight = (movie_ratings[movie_id] / 5.0) ** 2

            for g1 in genres:
                for g2 in genres:
                    if g1 != g2:
                        cooccurrence[g1][g2] += rating_weight

        # Calculate genre frequencies
        genre_counts = defaultdict(int)
        for genres in movie_genres.values():
            for genre in genres:
                genre_counts[genre] += 1

        # Calculate similarities
        similarities = {}
        for g1 in all_genres:
            similarities[g1] = []
            norm1 = np.sqrt(genre_counts[g1])

            for g2, cooc in cooccurrence[g1].items():
                if g1 != g2:
                    norm2 = np.sqrt(genre_counts[g2])
                    if norm1 and norm2:
                        sim = cooc / (norm1 * norm2)
                        if sim > 0.3:  # Similarity threshold
                            similarities[g1].append((g2, float(sim)))

            # Keep top 3 similar genres
            similarities[g1] = sorted(
                similarities[g1], key=lambda x: x[1], reverse=True
            )[:3]

        return similarities, all_genres

    def get_genre_preferences(self, user_history: pd.DataFrame) -> Dict:
        """Get genre preferences with caching."""
        # Check cache validity
        current_key = self._get_cache_key(user_history)
        if current_key == self.cache_key:
            return self.genre_cache

        # Calculate genre statistics
        genre_stats = self._vectorize_genre_stats(user_history)

        # Calculate similarities only if needed
        if current_key != self.cache_key:
            similarities, all_genres = self._fast_similarity_calculation(user_history)
            self.similarity_cache = similarities

        # Process preferences
        preferences = {
            "liked_genres": [],
            "disliked_genres": [],
            "neutral_genres": [],
            "genre_scores": {},
            "similar_genres": {},
        }

        # Classify genres
        for genre, stats in genre_stats.items():
            # Calculate genre score
            base_score = stats["weighted_avg"] - 3.0
            confidence = stats["confidence"]
            reliability = 1 - (stats["rating_std"] / 5.0)

            genre_score = base_score * confidence * reliability
            preferences["genre_scores"][genre] = genre_score

            # Classify
            if genre_score > 0.5:
                preferences["liked_genres"].append(genre)
            elif genre_score < -0.5:
                preferences["disliked_genres"].append(genre)
            else:
                preferences["neutral_genres"].append(genre)

            # Add similar genres from cache
            if genre in self.similarity_cache:
                preferences["similar_genres"][genre] = [
                    {"genre": g, "similarity": s}
                    for g, s in self.similarity_cache[genre]
                ]

        # Update cache
        self.genre_cache = preferences
        self.cache_key = current_key

        return preferences

    def generate_genre_feature_text(self, preferences: Dict) -> str:
        """Generate concise text description of genre preferences."""
        parts = []

        # Add top liked genres
        if preferences["liked_genres"]:
            top_genres = sorted(
                preferences["liked_genres"],
                key=lambda g: preferences["genre_scores"][g],
                reverse=True,
            )[:3]

            scores = [f"{g} ({preferences['genre_scores'][g]:.2f})" for g in top_genres]
            parts.append(f"Top genres: {', '.join(scores)}")

            # Add one similar genre for each top genre
            for genre in top_genres:
                if (
                    genre in preferences["similar_genres"]
                    and preferences["similar_genres"][genre]
                ):
                    similar = preferences["similar_genres"][genre][0]
                    parts.append(f"Similar to {genre}: {similar['genre']}")

        # Add main disliked genres
        if preferences["disliked_genres"]:
            bottom_genres = sorted(
                preferences["disliked_genres"],
                key=lambda g: preferences["genre_scores"][g],
            )[:2]

            scores = [
                f"{g} ({preferences['genre_scores'][g]:.2f})" for g in bottom_genres
            ]
            parts.append(f"Less preferred: {', '.join(scores)}")

        return "\n".join(parts)

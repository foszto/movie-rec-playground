import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.serialization import safe_globals

from src.configs.logging_config import setup_logging
from src.configs.model_config import HybridConfig
from src.models.hybrid import HybridRecommender
from src.utils.data_io import load_preprocessed_data

# Load environment variables
load_dotenv()


# Configuration from environment variables
class Config:
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")  # nosec B104 , never goes to production
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "models/small/final_model.pt")
    DATA_DIR = os.getenv("DATA_DIR", "data/processed/small")

    # Recommendation Settings
    TOP_K_RECOMMENDATIONS = int(os.getenv("TOP_K_RECOMMENDATIONS", 5))
    MIN_RATING_THRESHOLD = float(os.getenv("MIN_RATING_THRESHOLD", 4.0))
    RECENT_DAYS_THRESHOLD = int(os.getenv("RECENT_DAYS_THRESHOLD", 90))

    # Model Settings
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Security
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

    # LLM Model Settings
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "all-MiniLM-L6-v2")
    LLM_EMBEDDING_DIM = int(os.getenv("LLM_EMBEDDING_DIM", 64))
    USE_CACHED_EMBEDDINGS = os.getenv("USE_CACHED_EMBEDDINGS", "True").lower() == "true"


# Initialize FastAPI app
app = FastAPI(title="Enhanced Movie Recommender API", debug=Config.DEBUG)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class MovieBase(BaseModel):
    id: int
    title: str
    genres: List[str]


class RatedMovie(MovieBase):
    rating: float
    timestamp: str
    ratingCount: Optional[int] = None
    similarMovies: Optional[List[str]] = None


class RecommendedMovie(MovieBase):
    predictedRating: float
    confidence: Optional[float] = None
    reason: Optional[str] = None


class UserProfile(BaseModel):
    favoriteGenres: List[str]
    averageRating: float
    totalRatings: int
    ratingDistribution: Dict[str, int]
    recentActivity: bool


class RecommendationResponse(BaseModel):
    userProfile: UserProfile
    topRatedMovies: List[RatedMovie]
    recentFavorites: List[RatedMovie]
    recommendations: List[RecommendedMovie]


# Global variables
model = None
movie_info_dict = None
user_history_dict = None
logger = logging.getLogger(__name__)


def calculate_user_profile(user_data) -> UserProfile:
    """Calculate detailed user profile from rating history."""
    recent_threshold = datetime.now() - timedelta(days=90)

    # Calculate rating distribution
    ratings = user_data["rating"].value_counts().to_dict()
    rating_dist = {str(k): int(v) for k, v in ratings.items()}

    # Calculate favorite genres
    genre_scores = {}
    for _, row in user_data.iterrows():
        movie_id = int(row["movieId"])
        if movie_id in movie_info_dict:
            movie = movie_info_dict[movie_id]
            rating_weight = row["rating"] / 5.0  # Normalize rating to 0-1

            for genre in movie.get("genres", []):
                if isinstance(genre, str):
                    genre_scores[genre] = genre_scores.get(genre, 0) + rating_weight

    # Sort genres by score
    favorite_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]  # Top 5 genres

    # Check recent activity
    latest_rating = datetime.fromtimestamp(user_data["timestamp"].max())
    is_active = latest_rating > recent_threshold

    return UserProfile(
        favoriteGenres=[genre for genre, _ in favorite_genres],
        averageRating=float(user_data["rating"].mean()),
        totalRatings=len(user_data),
        ratingDistribution=rating_dist,
        recentActivity=is_active,
    )


def get_top_rated_movies(user_data, limit: int = None) -> List[RatedMovie]:
    """Get user's highest rated movies with additional context."""
    limit = limit or Config.TOP_K_RECOMMENDATIONS
    rated_movies = []

    for _, row in user_data.iterrows():
        movie_id = int(row["movieId"])
        if movie_id in movie_info_dict:
            if row["rating"] >= Config.MIN_RATING_THRESHOLD:
                movie_info = movie_info_dict[movie_id]
                genres = movie_info.get("genres", [])
                if isinstance(genres, str):
                    genres = [g.strip() for g in genres.split("|")]

                rated_movies.append(
                    {
                        "id": movie_id,
                        "title": movie_info.get("title", "Unknown"),
                        "genres": genres,
                        "rating": float(row["rating"]),
                        "timestamp": datetime.fromtimestamp(
                            row["timestamp"]
                        ).isoformat(),
                        "ratingCount": None,
                        "similarMovies": [],
                    }
                )

    rated_movies.sort(key=lambda x: (-x["rating"], x["timestamp"]), reverse=False)
    return rated_movies[:limit]


def get_recent_favorites(user_data, limit: int = None) -> List[RatedMovie]:
    """Get user's recent favorite movies."""
    limit = limit or Config.TOP_K_RECOMMENDATIONS
    cutoff_date = datetime.now() - timedelta(days=Config.RECENT_DAYS_THRESHOLD)

    recent_favorites = []
    for _, row in user_data.iterrows():
        rating_date = datetime.fromtimestamp(row["timestamp"])
        if rating_date > cutoff_date and row["rating"] >= Config.MIN_RATING_THRESHOLD:
            movie_id = int(row["movieId"])
            if movie_id in movie_info_dict:
                movie_info = movie_info_dict[movie_id]
                genres = movie_info.get("genres", [])
                if isinstance(genres, str):
                    genres = [g.strip() for g in genres.split("|")]

                recent_favorites.append(
                    {
                        "id": movie_id,
                        "title": movie_info.get("title", "Unknown"),
                        "genres": genres,
                        "rating": float(row["rating"]),
                        "timestamp": rating_date.isoformat(),
                    }
                )

    recent_favorites.sort(key=lambda x: x["timestamp"], reverse=True)
    return recent_favorites[:limit]


async def generate_recommendations(
    user_id: int, limit: int = 10
) -> List[RecommendedMovie]:
    """Generate personalized recommendations with explanations."""
    recommendations = await model.get_top_n_recommendations(
        user_id,
        movie_info_dict,
        user_history_dict,
        n=limit * 4,  # More recommendations to filter out the best ones
        exclude_watched=True,
    )

    # First, get user's favorite genres
    user_data = user_history_dict[user_id]
    genre_ratings = {}
    for _, row in user_data.iterrows():
        if row["rating"] >= 3.5:
            movie_id = int(row["movieId"])
            if movie_id in movie_info_dict:
                for genre in movie_info_dict[movie_id].get("genres", []):
                    if isinstance(genre, str):
                        genre_ratings[genre] = genre_ratings.get(genre, 0) + 1

    favorite_genres = sorted(genre_ratings.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]
    favorite_genres = [genre for genre, _ in favorite_genres]

    result = []
    remaining_movies = []

    # Filter recommendations based on favorite genres
    for movie_id, predicted_rating in recommendations:
        movie_info = movie_info_dict[movie_id]
        genres = movie_info.get("genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split("|")]

        matching_genres = [g for g in genres if g in favorite_genres]
        movie_data = {
            "id": movie_id,
            "title": movie_info.get("title", "Unknown"),
            "genres": genres,
            "predictedRating": float(predicted_rating),
            "matching_count": len(matching_genres),
            "matching_genres": matching_genres,
        }

        if len(matching_genres) >= 2:
            movie_data["confidence"] = min(1.0, 0.6 + 0.1 * len(matching_genres))
            movie_data["reason"] = (
                f"This matches your favorite genres: {', '.join(matching_genres)}"
            )
            result.append(movie_data)
        else:
            remaining_movies.append(movie_data)

        if len(result) >= limit:
            break

    # If we still don't have enough recommendations, add one-match movies
    if len(result) < limit:
        one_match_movies = [m for m in remaining_movies if m["matching_count"] == 1]
        for movie in one_match_movies:
            movie["confidence"] = 0.6
            movie["reason"] = (
                f"This matches one of your favorite genres: {movie['matching_genres'][0]}"
            )
            result.append(movie)
            if len(result) >= limit:
                break

    # If we still don't have enough recommendations, add non-matching movies
    if len(result) < limit:
        no_match_movies = [m for m in remaining_movies if m["matching_count"] == 0]
        for movie in no_match_movies:
            movie["confidence"] = 0.5
            movie["reason"] = "Try something completely new!"
            result.append(movie)
            if len(result) >= limit:
                break

    # Clean up the result
    for movie in result:
        movie.pop("matching_count", None)
        movie.pop("matching_genres", None)

    return result


@app.on_event("startup")
async def load_model():
    global model, movie_info_dict, user_history_dict

    try:
        # Setup logging
        setup_logging()

        if not Path(Config.MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

        # Load checkpoint with fallback
        try:
            checkpoint = torch.load(
                Config.MODEL_PATH,
                map_location=torch.device(Config.DEVICE),
                weights_only=True,
            )
            logger.info("Model loaded with weights_only=True")
        except Exception:
            logger.warning(
                "Could not load with weights_only=True, trying with weights_only=False"
            )
            with safe_globals([HybridConfig]):
                checkpoint = torch.load(  # nosec B614 - Trusted model checkpoint, created by our training pipeline
                    Config.MODEL_PATH,
                    map_location=torch.device(Config.DEVICE),
                    weights_only=False,
                )
            logger.info("Model loaded with weights_only=False")

        # Update config with environment settings
        config_dict = checkpoint["config"]
        config_dict.update(
            {
                "device": Config.DEVICE,
                "llm_model_name": Config.LLM_MODEL_NAME,
                "llm_embedding_dim": Config.LLM_EMBEDDING_DIM,
            }
        )

        config = HybridConfig(**config_dict)

        # Initialize and load model
        model = HybridRecommender(config)
        model.load(Config.MODEL_PATH)
        logger.info("Model initialized successfully")

        # Load data
        logger.info("Loading preprocessed data...")
        _, _, _, user_history_dict, movie_info_dict, _, _ = load_preprocessed_data(
            Config.DATA_DIR
        )
        logger.info("Data loaded successfully")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.get("/api/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, limit: int = Config.TOP_K_RECOMMENDATIONS):
    try:
        if user_id not in user_history_dict:
            raise HTTPException(status_code=404, detail="User not found")

        user_data = user_history_dict[user_id]

        # Generate all components in parallel
        recommendations = await generate_recommendations(user_id, limit)
        top_rated = get_top_rated_movies(user_data, limit)
        recent_favorites = get_recent_favorites(user_data, limit)
        user_profile = calculate_user_profile(user_data)

        return {
            "userProfile": user_profile,
            "topRatedMovies": top_rated,
            "recentFavorites": recent_favorites,
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error generating recommendations"
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="debug" if Config.DEBUG else "info",
    )

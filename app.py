from fastapi import FastAPI, HTTPException
from torch.serialization import safe_globals
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import logging
from pathlib import Path
import asyncio
from datetime import datetime

from src.models.hybrid import HybridRecommender
from src.configs.model_config import HybridConfig
from src.utils.data_io import load_preprocessed_data
from src.configs.logging_config import setup_logging

# Initialize FastAPI app
app = FastAPI(title="Movie Recommender API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
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

class RecommendedMovie(MovieBase):
    predictedRating: float

class RecommendationResponse(BaseModel):
    userHistory: List[RatedMovie]
    recommendations: List[RecommendedMovie]

# Global variables for model and data
model = None
movie_info_dict = None
user_history_dict = None

@app.on_event("startup")
async def load_model():
    global model, movie_info_dict, user_history_dict
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Load model
        model_path = "models/high/final_model.pt"  # Update with your model path
        data_dir = "data/processed_high"  # Update with your data directory
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load checkpoint with fallback
        try:
            # First try with weights_only=True
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            logger.info("Model loaded with weights_only=True")
        except Exception as e:
            logger.warning("Could not load with weights_only=True, trying with weights_only=False")
            # If that fails, try with weights_only=False
            with safe_globals([HybridConfig]):
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            logger.info("Model loaded with weights_only=False")
            
        config = HybridConfig(**checkpoint['config'])
        
        # Initialize and load model
        model = HybridRecommender(config)
        model.load(model_path)
        logger.info("Model loaded successfully")
        
        # Load data
        logger.info("Loading preprocessed data...")
        _, _, _, user_history_dict, movie_info_dict, _, _ = load_preprocessed_data(data_dir)
        logger.info("Data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

def format_movie_info(movie_id: int, movie_info: Dict, rating: Optional[float] = None, 
                     timestamp: Optional[str] = None) -> Dict:
    """Helper function to format movie information."""
    genres = movie_info.get('genres', [])
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split('|')]
    
    movie_dict = {
        "id": movie_id,
        "title": movie_info.get('title', 'Unknown Title'),
        "genres": genres
    }
    
    if rating is not None:
        movie_dict["rating"] = float(rating)
    if timestamp is not None:
        movie_dict["timestamp"] = timestamp
        
    return movie_dict

@app.get("/api/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, limit: int = 5):
    try:
        # Validate user exists
        if user_id not in user_history_dict:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user history
        user_data = user_history_dict[user_id]
        
        # Format user history
        history = []
        for _, row in user_data.iterrows():
            movie_id = int(row['movieId'])
            if movie_id in movie_info_dict:
                movie = format_movie_info(
                    movie_id,
                    movie_info_dict[movie_id],
                    rating=row['rating'],
                    timestamp=datetime.fromtimestamp(row['timestamp']).isoformat()
                )
                history.append(movie)
        
        # Sort history by timestamp (newest first) and limit
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        history = history[:limit]
        
        # Get recommendations
        recommendations = await model.get_top_n_recommendations(
            user_id,
            movie_info_dict,
            user_history_dict,
            n=limit,
            exclude_watched=True
        )
        
        # Format recommendations
        recommended_movies = []
        for movie_id, predicted_rating in recommendations:
            if movie_id in movie_info_dict:
                movie = format_movie_info(movie_id, movie_info_dict[movie_id])
                movie["predictedRating"] = float(predicted_rating)
                recommended_movies.append(movie)
        
        return {
            "userHistory": history,
            "recommendations": recommended_movies
        }
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
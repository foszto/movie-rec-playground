# src/cli/recommend.py
import click
import logging
import torch
import asyncio
from pathlib import Path
from src.configs.logging_config import setup_logging
from src.models.hybrid import HybridRecommender
from src.utils.data_io import load_preprocessed_data

@click.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model')
@click.option('--user-id', type=int, required=True,
              help='User ID to generate recommendations for')
@click.option('--n-recommendations', type=int, default=10,
              help='Number of recommendations to generate')
async def recommend(model_path: str, user_id: int, n_recommendations: int):
    """Generate recommendations for a specific user."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load model and data
        logger.info("Loading model and data...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = HybridRecommender(checkpoint['config'])
        model.load(model_path)

        # Load user history and movie info from the data directory
        data_dir = Path(model_path).parent.parent  # Feltételezzük, hogy a modell mellett van a data
        _, _, user_history_dict, movie_info_dict = load_preprocessed_data(data_dir)

        # Get user history
        if user_id not in user_history_dict:
            logger.error(f"User {user_id} not found in the dataset.")
            return

        user_history = user_history_dict[user_id]
        watched_movies = set(user_history['movieId'])

        # Generate predictions for all unwatched movies
        all_movies = set(movie_info_dict.keys())
        unwatched_movies = list(all_movies - watched_movies)

        if not unwatched_movies:
            logger.warning(f"User {user_id} has watched all available movies!")
            return

        user_ids = torch.tensor([user_id] * len(unwatched_movies))
        item_ids = torch.tensor(unwatched_movies)

        # Get predictions
        logger.info("Generating recommendations...")
        predictions = await model.predict(
            user_ids,
            item_ids,
            movie_info_dict,
            user_history_dict
        )

        # Sort and get top recommendations
        movie_scores = list(zip(unwatched_movies, predictions.cpu().numpy()))
        movie_scores.sort(key=lambda x: x[1], reverse=True)

        # Print recommendations
        logger.info(f"\nTop {n_recommendations} recommendations for user {user_id}:")
        print(f"\nUser {user_id}'s rating history:")
        print(f"- Total movies rated: {len(watched_movies)}")
        print(f"- Average rating: {user_history['rating'].mean():.2f}")
        print(f"\nTop {n_recommendations} recommendations:")

        for movie_id, score in movie_scores[:n_recommendations]:
            movie_info = movie_info_dict[movie_id]
            print(f"\n- {movie_info['title']}")
            print(f"  Score: {score:.2f}")
            print(f"  Genres: {', '.join(movie_info['genres'])}")

    except Exception as e:
        logger.error(f"Error during recommendation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    asyncio.run(recommend())
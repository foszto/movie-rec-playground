# src/cli/recommend.py
import click
import logging
import torch
import asyncio
from pathlib import Path
from src.configs.logging_config import setup_logging
from src.models.hybrid import HybridRecommender
from src.utils.data_io import load_preprocessed_data
from src.configs.model_config import HybridConfig
from torch.serialization import add_safe_globals, safe_globals

@click.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model')
@click.option('--data-dir', type=click.Path(exists=True),
              help='Path to data directory containing preprocessed files. If not provided, will try to infer from model path.')
@click.option('--user-id', type=int, required=True,
              help='User ID to generate recommendations for')
@click.option('--n-recommendations', type=int, default=10,
              help='Number of recommendations to generate')
@click.option('--include-watched/--exclude-watched', default=False,
              help='Include or exclude already watched movies in recommendations')
def recommend(model_path: str, data_dir: str, user_id: int, n_recommendations: int, include_watched: bool):
    """Generate recommendations for a specific user."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load model and data
        logger.info("Loading model and data...")
        
        # Add HybridConfig to safe globals
        add_safe_globals([HybridConfig])
        
        try:
            # First try with weights_only=True
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except Exception as e:
            logger.warning("Could not load with weights_only=True, trying with weights_only=False")
            # If that fails, try with weights_only=False
            with safe_globals([HybridConfig]):
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        config = HybridConfig(**checkpoint['config'])
        model = HybridRecommender(config)
        model.load(model_path)

        # Determine data directory
        if data_dir is None:
            data_dir = Path(model_path).parent.parent
            logger.info(f"No data directory provided, using inferred path: {data_dir}")
        else:
            data_dir = Path(data_dir)
            logger.info(f"Using provided data directory: {data_dir}")

        # Load user history and movie info
        logger.info("Loading preprocessed data...")
        try:
            _, _, _, user_history_dict, movie_info_dict, _, _ = load_preprocessed_data(data_dir)
        except FileNotFoundError as e:
            logger.error(f"Could not find required data files in {data_dir}. Please check if the data directory contains preprocessed files.")
            raise click.UsageError(f"Data files not found in directory: {data_dir}") from e

        # Get user history
        if user_id not in user_history_dict:
            logger.error(f"User {user_id} not found in the dataset.")
            return

        user_history = user_history_dict[user_id]
        watched_movies = set(user_history['movieId'])
        
        # Print user history summary
        print(f"\nUser {user_id}'s viewing history:")
        print(f"- Total movies rated: {len(watched_movies)}")
        print(f"- Average rating: {user_history['rating'].mean():.2f}")
        
        # Show some recently rated movies
        print("\nRecent ratings:")
        recent_ratings = sorted(zip(user_history['movieId'], user_history['rating'], user_history['timestamp']), 
                              key=lambda x: x[2], reverse=True)[:5]
        for movie_id, rating, _ in recent_ratings:
            movie_info = movie_info_dict[movie_id]
            print(f"- {movie_info['title']}: {rating:.1f}/5.0")

        # Generate candidate movies list
        all_movies = set(movie_info_dict.keys())
        if include_watched:
            candidate_movies = list(all_movies)
            logger.info("Including already watched movies in recommendations")
        else:
            candidate_movies = list(all_movies - watched_movies)
            logger.info("Excluding already watched movies from recommendations")

        if not candidate_movies:
            logger.warning(f"No {'unwatched ' if not include_watched else ''}movies available for recommendations!")
            return

        user_ids = torch.tensor([user_id] * len(candidate_movies))
        item_ids = torch.tensor(candidate_movies)

        # Get predictions
        logger.info("Generating recommendations...")
        predictions = asyncio.run(model.predict(
            user_ids,
            item_ids,
            movie_info_dict,
            user_history_dict
        ))

        # Sort and get top recommendations
        movie_scores = list(zip(candidate_movies, predictions.cpu().numpy()))
        movie_scores.sort(key=lambda x: x[1], reverse=True)

        # Print recommendations
        print(f"\nTop {n_recommendations} recommendations:")
        
        for movie_id, score in movie_scores[:n_recommendations]:
            movie_info = movie_info_dict[movie_id]
            status = " [Previously rated]" if movie_id in watched_movies else " [New]"
            if movie_id in watched_movies:
                previous_rating = user_history.loc[user_history['movieId'] == movie_id, 'rating'].iloc[0]
                status = f" [Previously rated: {previous_rating:.1f}/5.0]"
            
            print(f"\n- {movie_info['title']}{status}")
            print(f"  Predicted score: {score:.2f}")
            print(f"  Genres: {', '.join(movie_info['genres'])}")

        # Print recommendation statistics
        recommended_movies = set(movie_id for movie_id, _ in movie_scores[:n_recommendations])
        new_recommendations = recommended_movies - watched_movies
        print(f"\nRecommendation statistics:")
        print(f"- New movies: {len(new_recommendations)}/{n_recommendations}")
        print(f"- Previously rated: {n_recommendations - len(new_recommendations)}/{n_recommendations}")

    except Exception as e:
        logger.error(f"Error during recommendation: {str(e)}", exc_info=True)
        raise
# Movie Recommendation System

A hybrid movie recommendation system combining collaborative filtering with LLM (Language Learning Model) features. The system uses the MovieLens dataset and incorporates both traditional recommendation techniques and modern language models for enhanced recommendations.

## Features

- Hybrid recommendation system combining:
  - Collaborative filtering
  - Content-based filtering using LLM features
  - Diversity-aware recommendations
- GPU-accelerated training and inference
- Comprehensive preprocessing pipeline
- REST API for serving recommendations
- Web interface for testing recommendations

## Project Structure

```
.
├── src/
│   ├── cli/           # Command line interface tools
│   ├── configs/       # Configuration files
│   ├── data/         # Data processing modules
│   ├── llm/          # LLM integration
│   ├── models/       # Model implementations
│   └── utils/        # Utility functions
├── data/
│   ├── raw/          # Raw MovieLens dataset
│   └── processed/    # Preprocessed data
├── models/           # Saved model checkpoints
├── frontend/         # Web interface
└── tests/           # Test suite
```

## Pre-trained Models

Three variants of the model are available, each trained on differently preprocessed versions of the MovieLens dataset:

### Tiny Model

Trained on a highly filtered dataset with strict rating requirements:

- Minimum user ratings: 500 (significantly higher than default 5)
- Minimum movie ratings: 500 (significantly higher than default 5)
- Minimum tags per item: 3 (default)
  This variant provides recommendations based on the most active users and most-rated movies only, resulting in a smaller but highly reliable dataset.

### Small Model

Uses moderately filtered data:

- Minimum user ratings: 500 (higher than default 5)
- Minimum movie ratings: 5 (default)
- Minimum tags per item: 3 (default)
  This variant balances dataset size with quality by keeping the higher user rating requirement while using default thresholds for movies and tags.

### Large Model

Uses minimally filtered data with all default thresholds:

- Minimum user ratings: 5 (default)
- Minimum movie ratings: 5 (default)
- Minimum tags per item: 3 (default)
  This variant provides recommendations based on the broadest possible dataset, including casual users and less popular movies.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker and Docker Compose

## Environment Setup

1. Clone the repository
2. Create a `.env` file from .env.dist:

## Quick Start

1. Start the services:

```bash
docker-compose up -d
```

2. Access the services:

- Frontend: localhost:3000

3. Run commands in the dev container:

   ```bash
   docker-compose exec dev bash
   ```

4. Stop the services:
   ```bash
   docker-compose down
   ```

## Data Pipeline

### 1. Preprocessing

Process the raw MovieLens dataset:

```bash
python main.py preprocess \
  --data-dir DATA_DIR \
  --output-dir OUTPUT_DIR \
  --min-user-ratings MIN_USER_RATINGS \
  --min-movie-ratings MIN_MOVIE_RATINGS \
  --min-tags-per-item MIN_TAGS_PER_ITEM \
  --log-level {DEBUG,INFO,WARNING,ERROR} \
  --log-file LOG_FILE
```

Options:

- `--data-dir`: Directory containing MovieLens dataset files (required)
- `--output-dir`: Directory to save preprocessed data (required)
- `--min-user-ratings`: Minimum number of ratings per user (default: 5)
- `--min-movie-ratings`: Minimum number of ratings per movie (default: 5)
- `--min-tags-per-item`: Minimum number of tags per item (default: 3)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
- `--log-file`: Optional log file path

### 2. Training

Train the hybrid recommendation model:

```bash
python main.py train \
  --data-dir DATA_DIR \
  --output-dir OUTPUT_DIR \
  --config-path CONFIG_PATH
```

Options:

- `--data-dir`: Directory containing preprocessed data (required)
- `--output-dir`: Directory to save model and results (required)
- `--config-path`: Path to model configuration file (required)

The learning and validation phase of the first Epoch also starts slowly, gradually speeding up as the cache builds up, with speed increases of up to 1000x depending on hardware

### 3. Evaluation

Evaluate the trained model:

```bash
python main.py evaluate \
  --data-dir DATA_DIR \
  --model-path MODEL_PATH \
  --output-dir OUTPUT_DIR \
  --batch-size BATCH_SIZE
```

Options:

- `--data-dir`: Directory containing test data (required)
- `--model-path`: Path to trained model (required)
- `--output-dir`: Directory to save evaluation results (required)
- `--batch-size`: Batch size for evaluation (default: 128)

### 4. Generating Recommendations

Generate recommendations for a specific user:

```bash
python main.py recommend \
  --model-path MODEL_PATH \
  --data-dir DATA_DIR \
  --user-id USER_ID \
  --n-recommendations N \
  --include-watched/--exclude-watched
```

Options:

- `--model-path`: Path to trained model (required)
- `--data-dir`: Path to data directory containing preprocessed files (optional, will try to infer from model path)
- `--user-id`: User ID to generate recommendations for (required)
- `--n-recommendations`: Number of recommendations to generate (default: 10)
- `--include-watched/--exclude-watched`: Include or exclude already watched movies (default: exclude)

## API Endpoints

### Get Recommendations

```http
GET /api/recommendations/{user_id}
```

Parameters:

- `user_id`: Integer
- `limit`: Integer (optional, default=5)

Response includes:

- User profile
- Top rated movies
- Recent favorites
- Personalized recommendations

## Model Architecture

The system uses a hybrid architecture combining:

1. Collaborative Filtering:

   - User and item embeddings
   - Global bias terms
   - Batch normalization

2. LLM Features:

   - Sentence transformer embeddings
   - Cross-attention mechanism
   - Genre and tag processing

3. Diversity-Aware Loss:
   - Rating prediction loss
   - Diversity penalty
   - Rating distribution loss

## Dataset

The project includes the MovieLens dataset with:

- movies.csv
- ratings.csv
- tags.csv

## Development

Run tests:

```bash
docker-compose exec dev pytest tests/
```

Code formatting:

```bash
docker-compose exec dev black .
docker-compose exec dev isort .
```

## License

MIT

## Contributors

Feel free to contribute to this project by submitting issues or pull requests.

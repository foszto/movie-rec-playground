# Recipe Recommendation System

This project implements a personalized recipe recommendation system using the Food.com dataset.

## Project Structure

Main components:

- Data preprocessing
- Collaborative filtering-based recommendations
- Content-based filtering
- LLM integration
- Hybrid recommendation system

## Prerequisites

- Docker and Docker Compose installed on your system

## Setup & Usage

1. **Start the Project**

   ```bash
   # Build and start the containers
   docker-compose up -d
   ```

## Development

The project includes 3 services:

1. **Development Container**
   - Used for running scripts and tools
   ```bash
   # Example: Run tests
   docker-compose exec dev pytest tests/
   ```
2. **Api Container**

   - Used for running api backend of recommendation model

3. **Frontend container**
   - Simple test user interface for recommendation

## Development Workflow

1. Start the services:

   ```bash
   docker-compose up -d
   ```

2. Access the frontend

   - Open localhost:3000

3. Run commands in the dev container:

   ```bash
   docker-compose exec dev bash
   ```

4. Stop the services:
   ```bash
   docker-compose down
   ```

## Dataset

The project includes the MovieLens dataset with:

- movies.csv
- ratings.csv
- tags.csv

## Preprocess

```bash
docker-compose exec dev bash
python main.py preprocess --data-dir data/raw --output-dir data/processed/small --min-user-ratings 500 --log-file logs/preprocess_log.txt
```

## Train

```bash
docker-compose exec dev bash
python main.py train --data-dir data/processed/small --output-dir models/small --config-path src/configs/model_config.yaml

python main.py train --data-dir data/processed/high --output-dir models/high --config-path src/configs/model_config.yaml
```

## Evaluate

```bash
docker-compose exec dev bash
python main.py evaluate --data-dir data/processed/small --model-path models/small --output-dir models/small/eval
```

## Recommend

```bash
docker-compose exec dev bash
python main.py recommend --data-dir data/processed --model-path output/final_model.pt --user-id 101 --include-watched
```

## License

MIT

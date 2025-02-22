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

2. **Open JupyterLab**
   - Navigate to http://localhost:8888 in your browser
   - Open notebooks/01_eda.ipynb to start the analysis

## Development

The project includes two services:

1. **Jupyter Service**
   - Access JupyterLab at http://localhost:8888
   - Used for interactive development and visualization

2. **Development Container**
   - Used for running scripts and tools
   ```bash
   # Example: Run tests
   docker-compose exec dev pytest tests/
   ```

## Development Workflow

1. Start the services:
   ```bash
   docker-compose up -d
   ```

2. Access JupyterLab for interactive development:
   - Open http://localhost:8888

3. Run commands in the dev container:
   ```bash
   docker-compose exec dev bash
   ```

4. Stop the services:
   ```bash
   docker-compose down
   ```
## Dataset

The project includes the Food.com dataset with:
- RAW_recipes.csv: Contains recipe details
- RAW_interactions.csv: Contains user-recipe interactions

## Preprocess
   ```bash
   docker-compose exec dev bash
   python main.py preprocess --data-dir data/raw --output-dir data/processed --min-user-ratings 500 --log-file logs/preprocess_log.txt
   ```

## Train

   ```bash
   docker-compose exec dev bash
   python main.py train --data-dir data/processed --output-dir models --config-path src/configs/model_config.yaml
   ```

## Evaluate

   ```bash
   docker-compose exec dev bash
   python main.py evaluate --data-dir data/processed --model-path models --output-dir output
   ```

## Recommend

   ```bash
   docker-compose exec dev bash
   python main.py recommend --data-dir data/processed --model-path output/final_model.pt --user-id 101 --include-watched
   ```
## License

MIT
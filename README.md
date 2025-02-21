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

## License

MIT
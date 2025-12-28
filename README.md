# Penguin Species Prediction Project

This project implements an end-to-end machine learning pipeline using Metaflow for training and MLflow for experiment tracking. It also provides a FastAPI-based inference API to serve real-time predictions for penguin species.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Metaflow Pipeline](#running-the-metaflow-pipeline)
- [Running the Prediction API](#running-the-prediction-api)
- [Making Predictions](#making-predictions)
- [MLflow UI](#mlflow-ui)

## Project Structure

```
.
├── data/
│   └── penguins.csv          # Raw dataset
├── notebooks/
│   └── 0_data_exploration.ipynb # Jupyter notebook for initial data exploration
├── src/
│   ├── api.py                # FastAPI application for inference
│   └── pipelines/
│       ├── __init__.py       # Makes 'pipelines' a Python package
│       ├── cleaning.py       # Data cleaning functions
│       ├── config.py         # Project configuration (e.g., column names, hyperparameters)
│       ├── data.py           # Data loading, splitting, and encoding functions
│       ├── features.py       # Feature engineering pipeline components
│       ├── model.py          # XGBoost model definition
│       ├── tracker.py        # MLflow tracking wrapper class
│       └── trainer.py        # Model training and evaluation logic
├── tests/                    # Unit tests for pipeline components
├── penguin_flow.py           # The main Metaflow training pipeline
├── pyproject.toml            # Project dependencies and metadata
└── README.md                 # This documentation file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd penguin_school
    ```

2.  **Create and activate a virtual environment (recommended):**
    This project uses `uv` for dependency management.
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```

## Running the Metaflow Pipeline

The `penguin_flow.py` script orchestrates the entire ML pipeline:
- Loads and cleans the data.
- Splits data into training and testing sets.
- Performs feature engineering and target encoding.
- Trains an XGBoost model with cross-validation.
- Logs all parameters, metrics, the feature pipeline, and the trained model to MLflow.
- Retrains the best model on the full dataset.

To run the Metaflow pipeline:

```bash
python penguin_flow.py run
```

This command will execute the pipeline, and you will see output in your terminal detailing the progress of each step.

## Running the Prediction API

The FastAPI application (`src/api.py`) loads the latest trained model and feature pipeline from MLflow and exposes a `/predict` endpoint for inference.

1.  **Start the MLflow UI server (if not already running):**
    The API needs to connect to the MLflow tracking server to load the model artifacts.
    ```bash
    mlflow ui &
    ```
    Note the `&` at the end to run it in the background. You can check the PID with `jobs` and kill it later with `kill <PID>`.

2.  **Start the FastAPI server:**
    ```bash
    uvicorn src.api:app --reload
    ```
    The `--reload` flag is useful for development as it restarts the server whenever code changes are detected.

The API will be accessible at `http://127.0.0.1:8000`.

## Making Predictions

Once the FastAPI server is running, you can interact with it.

1.  **Access the interactive API documentation:**
    Open your web browser and go to `http://127.0.0.1:8000/docs`. Here you can see the available endpoints and test them directly.

2.  **Send a prediction request using `curl`:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d 
    {
      "bill_length_mm": 49.9,
      "bill_depth_mm": 16.1,
      "flipper_length_mm": 213,
      "body_mass_g": 5400,
      "sex": "male"
    }
    ```
    This will return a JSON response with the predicted penguin species.

## MLflow UI

You can access the MLflow UI by navigating to `http://127.0.0.1:5000` in your web browser. Here, you can explore:
- All recorded experiments and runs.
- Logged parameters, metrics, and artifacts (models, feature pipelines, label mappings).
- Compare different runs to analyze model performance.

```
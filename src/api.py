import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

# Define the input data schema using Pydantic.
# This ensures that requests to the /predict endpoint will be validated
# against this schema, preventing malformed data from crashing the application.
class PenguinData(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str

# Initialize the FastAPI app. This is the main entry point for the API.
app = FastAPI(title="Penguin Species Predictor API", version="1.0")

# --- MLflow Model Loading ---
# On startup, the API connects to the MLflow tracking server to find and load the
# necessary artifacts from the latest successful training run. These artifacts
# are then held in memory to serve predictions quickly.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Find the latest run from the 'penguin-school' experiment (assumes experiment ID '0')
try:
    runs = mlflow.search_runs(experiment_ids=["0"])
    latest_run_id = runs.iloc[0].run_id

    # Load the artifacts from the determined run
    feature_pipeline = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/feature_pipeline")
    model = mlflow.xgboost.load_model(f"runs:/{latest_run_id}/xgboost-model")
    label_map = mlflow.artifacts.load_dict(f"runs:/{latest_run_id}/label_map.json")
    
    # MLflow saves dict keys as strings, so convert them back to integers for lookup
    label_map = {int(k): v for k, v in label_map.items()}
    print(f"Successfully loaded artifacts from run_id: {latest_run_id}")

except Exception as e:
    print(f"Error loading MLflow artifacts: {e}")
    print("API will not be able to serve predictions.")
    feature_pipeline, model, label_map = None, None, None
# --- End of MLflow Model Loading ---

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Penguin Species Prediction API"}

@app.post("/predict")
def predict(data: PenguinData):
    """
    Makes a prediction on a single instance of penguin data.

    This endpoint takes a JSON object matching the PenguinData schema,
    processes it with the loaded feature pipeline, and returns the predicted
    penguin species.
    """
    if not all([feature_pipeline, model, label_map]):
        return {"error": "Model or artifacts not loaded. Check API server logs."}

    # Convert the input Pydantic model to a pandas DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Apply the same feature transformations used during training
    transformed_data = feature_pipeline.transform(df)

    # Get the numeric prediction from the model
    prediction_numeric = model.predict(transformed_data)
    
    # Map the numeric prediction back to its original string label
    prediction_label = label_map.get(prediction_numeric[0], "Unknown")

    return {"predicted_species": prediction_label}

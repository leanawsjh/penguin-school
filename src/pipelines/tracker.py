import mlflow
from mlflow.tracking import MlflowClient

class MLFlowTracker:
    """A wrapper class for simplifying MLflow experiment tracking interactions."""

    def __init__(self, experiment_name: str, tracking_uri: str):
        """
        Initializes the MLflow tracker.

        Sets the tracking URI and ensures the specified experiment exists.

        Args:
            experiment_name: The name of the MLflow experiment.
            tracking_uri: The URI of the MLflow tracking server.
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment = self._get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

    def _get_or_create_experiment(self, experiment_name: str):
        """

        Retrieves an existing MLflow experiment by name or creates it if it
        does not exist.

        Args:
            experiment_name: The name of the experiment.

        Returns:
            The MLflow experiment object.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found. Creating a new one.")
            return self.client.create_experiment(experiment_name)
        print(f"Found existing experiment '{experiment_name}'.")
        return experiment

    def start_run(self):
        """Starts a new MLflow run context."""
        return mlflow.start_run()

    def log_params(self, params: dict):
        """Logs a dictionary of parameters for the current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        """Logs a dictionary of metrics for the current run."""
        mlflow.log_metrics(metrics)

    def log_model(self, model, artifact_path: str):
        """
        Logs a trained model as an artifact.

        Args:
            model: The trained model object (e.g., an XGBoost model).
            artifact_path: The name to save the model artifact as.
        """
        mlflow.xgboost.log_model(model, artifact_path)

    def end_run(self):
        """Ends the current MLflow run."""
        mlflow.end_run()

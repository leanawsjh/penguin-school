from metaflow import FlowSpec, step
import numpy as np
import mlflow

# Import your project's configuration and data handling functions
from src.pipelines.config import TrainConfig
from src.pipelines.data import get_cleaned_data, split_data, encode_target
from src.pipelines.tracker import MLFlowTracker

# Import the rest of your pipeline components
from src.pipelines.features import (
    FeaturePipeline,
    NumericScaler,
    CategoricalEncoder,
)
from src.pipelines.model import XGBoostModel
from src.pipelines.trainer import Trainer


class PenguinFlow(FlowSpec):
    """
    A Metaflow pipeline for training a penguin species classification model.

    This flow orchestrates the end-to-end process of data loading, preprocessing,
    model training, evaluation, and artifact tracking with MLflow.
    """

    @step
    def start(self):
        """
        Initializes the run and loads, cleans, and splits the dataset.

        This step sets up the configuration, initializes the MLflow tracker,
        and prepares the training and testing datasets for the pipeline.
        The target variable is also label-encoded here.
        """
        print("Starting the flow...")
        self.config = TrainConfig()
        self.tracker = MLFlowTracker(
            experiment_name="penguin-school", tracking_uri="http://127.0.0.1:5000"
        )
        cleaned_df = get_cleaned_data(self.config)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            cleaned_df, self.config
        )
        self.y_train = encode_target(self.y_train)
        self.y_test = encode_target(self.y_test)
        print("Data loading, cleaning, and splitting complete.")
        self.next(self.train)

    @step
    def train(self):
        """
        Trains the model, performs cross-validation, and logs artifacts to MLflow.

        This step is the core of the pipeline. It initializes an MLflow run,
        builds a feature engineering pipeline, and trains an XGBoost model.
        It logs model parameters, cross-validation metrics, the final trained model,
        the feature pipeline, and the target label mapping as artifacts in MLflow.
        """
        print("Starting model training step...")

        with self.tracker.start_run():
            mlflow.set_tracking_uri(self.tracker.client.tracking_uri)
            
            # Define and fit the feature engineering pipeline
            feature_pipeline = FeaturePipeline(
                numeric_features=self.config.num_columns,
                categorical_features=self.config.cat_columns,
                numeric_scaler=NumericScaler(),
                categorical_encoder=CategoricalEncoder(),
            )
            feature_pipeline.fit(self.X_train)
            X_train_t = feature_pipeline.transform(self.X_train)

            # Log the fitted feature pipeline and label encoder to MLflow
            mlflow.sklearn.log_model(feature_pipeline, "feature_pipeline")
            le = self.y_train.attrs['le']
            labels = le.classes_
            label_map = {i: labels[i] for i in range(len(labels))}
            mlflow.log_dict(label_map, "label_map.json")

            # Initialize and configure the XGBoost model
            model = XGBoostModel(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='mlogloss',
            )

            # Log model hyperparameters
            self.tracker.log_params({
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "random_state": self.config.random_state,
                "learning_rate": 0.1,
            })

            # --- Cross-Validation Step ---
            print("Performing 5-fold cross-validation on the training data...")
            self.cv_scores = model.evaluate(X_train_t, self.y_train, cv=5)
            self.cv_mean_accuracy = np.mean(self.cv_scores)
            self.tracker.log_metrics({"cv_mean_accuracy": self.cv_mean_accuracy})
            print(f"Mean cross-validation accuracy: {self.cv_mean_accuracy:.4f}")
            # --- End of Cross-Validation ---

            # --- Final Model Training Step ---
            print("Training final model on the full training set...")
            trainer = Trainer(model=model, features=feature_pipeline)
            trainer.train(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
            )
            self.model = model.model
            self.metrics = trainer.get_metrics()
            
            # Log final metrics and the trained model
            self.tracker.log_metrics(self.metrics)
            self.tracker.log_model(self.model, "xgboost-model")
            
            # Store the run ID for the next step
            self.run_id = mlflow.active_run().info.run_id
            print(f"Final model evaluation on test set complete. Metrics: {self.metrics}")

        self.next(self.train_best_model)

    @step
    def train_best_model(self):
        """
        Loads the best model from the run and retrains it on the entire dataset.

        This step simulates a production workflow where the final chosen model is
        retrained on all available data to maximize its performance before deployment.
        """
        print("Training the best model on the whole dataset...")
        mlflow.set_tracking_uri(self.tracker.client.tracking_uri)
        best_model_uri = f"runs:/{self.run_id}/xgboost-model"
        feature_pipeline_uri = f"runs:/{self.run_id}/feature_pipeline"
        print(f"Loading best model from: {best_model_uri}")

        # Prepare the full dataset for retraining
        cleaned_df = get_cleaned_data(self.config)
        
        # Load the fitted feature pipeline from the MLflow run
        feature_pipeline = mlflow.sklearn.load_model(feature_pipeline_uri)
        
        X_t = feature_pipeline.transform(cleaned_df)
        y = cleaned_df[self.config.target]
        y = encode_target(y)

        # Load and retrain the model
        best_model = mlflow.xgboost.load_model(best_model_uri)
        best_model.fit(X_t, y)
        print("Best model trained on the whole dataset.")
        self.next(self.end)

    @step
    def end(self):
        """
        Ends the flow and prints a summary of the results.
        """
        print("\nPenguinFlow has completed successfully!")
        print(f"Mean Cross-Validation Accuracy: {self.cv_mean_accuracy:.4f}")
        print(f"Final Test Set Accuracy: {self.metrics['accuracy']:.4f}")


if __name__ == "__main__":
    PenguinFlow()

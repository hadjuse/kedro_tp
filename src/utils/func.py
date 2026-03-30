import mlflow
from mlflow import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()


def push_to_model_registry(registry_name: str, model_uri: str) -> str:
    """
    Pushes a model's version to the specified registry.
    """
    tracking_uri = os.getenv("MLFLOW_SERVER")
    if not tracking_uri:
        raise ValueError("MLFLOW_SERVER environment variable is not set")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    result = client.create_model_version(
        name=registry_name,
        source=model_uri,  # Pass directly, no runs:/ wrapping
    )

    return result.version


if __name__ == "__main__":
    print(
        push_to_model_registry(
            "model", "gs://bucket_s3_hadj_kedro/mlflow/1/models/m-f540a103497a4b3e9e6d81cda96b12a5/artifacts"
        )
    )

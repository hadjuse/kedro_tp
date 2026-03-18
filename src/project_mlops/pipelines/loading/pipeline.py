from kedro.pipeline import Pipeline, Node
from .nodes import load_csv_from_bucket


def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                load_csv_from_bucket,
                [
                    "params:gcp_project_id",
                    "params:gcs_bucket_name",
                    "params:gcs_primary_folder",
                ],
                "primary",
            )
        ]
    )

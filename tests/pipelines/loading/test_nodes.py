from project_mlops.pipelines.loading.nodes import load_csv_from_bucket


def test_load_csv_from_bucket(project_id, bucket_name, primary_folder):
    df = load_csv_from_bucket(
        gcp_project_id=project_id,
        gcs_bucket_name=bucket_name,
        gcs_primary_folder=primary_folder,
    )
    print(df.head())

import pandas as pd

# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
from project_mlops.pipelines.loading.nodes import load_csv_from_bucket


class TestKedroRun:
    # def test_kedro_run_no_pipeline(self):
    #     # This example test expects a pipeline run failure, since
    #     # the default project template contains no pipelines.
    #     bootstrap_project(Path.cwd())

    #     with pytest.raises(Exception) as excinfo:  # noqa
    #         with KedroSession.create(project_path=Path.cwd()) as session:
    #             result = session.run()
    #             assert result is None

    def test_load_csv_from_bucket(self, project_id, bucket_name, primary_folder):
        df = load_csv_from_bucket(
            gcp_project_id=project_id,
            gcs_bucket_name=bucket_name,
            gcs_primary_folder=primary_folder,
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 16
        assert "purchased" in df

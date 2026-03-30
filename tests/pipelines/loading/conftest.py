import pytest
from kedro.io import DataCatalog, MemoryDataset


@pytest.fixture(scope="module")
def project_id():
    return "arboreal-totem-444713-e9"


@pytest.fixture(scope="module")
def primary_folder():
    return " data-test.csv/"


@pytest.fixture(scope="module")
def bucket_name():
    return "bucket_s3_hadj_kedro"


@pytest.fixture(scope="module")
def catalog_test(project_id, primary_folder, bucket_name):
    catalog = DataCatalog(
        {
            "params:gcp_project_id": MemoryDataset(project_id),
            "params:gcs_primary_folder": MemoryDataset(primary_folder),
            "params:gcs_bucket_name": MemoryDataset(bucket_name),
        }
    )
    return catalog

import pytest
from project_mlops.pipelines.loading.nodes import load_csv_from_bucket
from project_mlops.pipelines.purchase_predict.nodes import encode_features
from kedro.io import DataCatalog, MemoryDataset


@pytest.fixture(scope="module")
def project_id():
    return "arboreal-totem-444713-e9"


@pytest.fixture(scope="module")
def primary_folder():
    return "primary/primary/"


@pytest.fixture(scope="module")
def bucket_name():
    return "bucket_s3_hadj_kedro"


@pytest.fixture(scope="module")
def test_ratio():
    return 0.3


@pytest.fixture(scope="module")
def dataset_encoded(dataset_not_encoded):
    return encode_features(dataset_not_encoded)["features"]


@pytest.fixture(scope="module")
def dataset_not_encoded(project_id, bucket_name, primary_folder):
    return load_csv_from_bucket(project_id, bucket_name, primary_folder)


@pytest.fixture(scope="module")
def catalog_test(dataset_not_encoded, test_ratio):
    catalog = DataCatalog(
        {
            "primary": MemoryDataset(dataset_not_encoded),
            "params:test_ratio": MemoryDataset(test_ratio),
        }
    )
    return catalog

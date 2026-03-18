import pytest


@pytest.fixture(scope="module")
def project_id():
    return "arboreal-totem-444713-e9"


@pytest.fixture(scope="module")
def primary_folder():
    return "primary/primary/"


@pytest.fixture(scope="module")
def bucket_name():
    return "bucket_s3_hadj_kedro"

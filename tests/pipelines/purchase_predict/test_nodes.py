from project_mlops.pipelines.purchase_predict.nodes import (
    encode_features,
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd

BALANCE_THRESHOLD = 0.05
MIN_SAMPLES = 5000


def test_encode_features(dataset_not_encoded):
    encoded: dict[str, pd.DataFrame | dict[str, LabelEncoder]] = encode_features(dataset_not_encoded)
    features_value = encoded["features"]
    assert isinstance(features_value, pd.DataFrame), "Expected DataFrame for features"
    df: pd.DataFrame = features_value  # Now type-safe

    # Checking column 'purchased' that all values are either 0 or 1
    assert df["purchased"].isin([0, 1]).all()
    # Checking that all columns are numeric
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df.dtypes[col])
    # Checking that we have enough samples
    assert df.shape[0] > MIN_SAMPLES
    # Checking that classes have at least BALANCE_THRESHOLD percent of data
    assert (df["purchased"].value_counts() / df.shape[0] > BALANCE_THRESHOLD).all()

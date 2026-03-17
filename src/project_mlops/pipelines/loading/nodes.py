import pandas as pd
import io
from google.cloud import storage


def load_csv_from_bucket(gcp_project_id: str, gcs_bucket_name: str, gcs_primary_folder: str) -> pd.DataFrame:
    client = storage.Client(project=gcp_project_id)
    bucket = client.bucket(gcs_bucket_name)

    blobs = list(bucket.list_blobs(prefix=gcs_primary_folder))

    # Debug : vérifie ce qui est trouvé
    print(f"Fichiers trouvés : {len(blobs)}")
    for b in blobs:
        print(f"  - {b.name}")

    frames = []
    for blob in blobs:
        # Ignore les dossiers (noms qui finissent par '/')
        if blob.name.endswith("/"):
            continue
        # Ignore les fichiers non CSV
        if not blob.name.endswith(".csv"):
            continue

        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
        frames.append(df)

    if not frames:
        raise ValueError(f"Aucun CSV trouvé dans gs://{gcs_bucket_name}/{gcs_primary_folder}")

    return pd.concat(frames, ignore_index=True)

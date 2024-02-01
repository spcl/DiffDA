from google.cloud import storage
from google.cloud.storage import Bucket
from pathlib import Path
import os

"""
Provides utilities to load and cache gcs buckets.
"""

def parse_file_parts(file_name) -> dict:
    return dict(part.split("-", 1) for part in file_name.split("_"))

def authenticate_bucket() -> Bucket:
    # @title Authenticate with Google Cloud Storage
    # TODO: Figure out how to access a public cloud bucket without authentication.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcs_key.json"
    gcs_client = storage.Client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    return gcs_bucket

def save_to_dir(blob, directory: Path, name: str) -> None:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    with open(directory / name, "wb") as f:
        f.write(blob.read())
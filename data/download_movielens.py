"""Download and extract the MovieLens 1M dataset."""

import os
import urllib.request
import zipfile

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DATA_DIR, "ml-1m.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-1m")


def download_and_extract():
    if os.path.exists(EXTRACT_DIR):
        print(f"Dataset already exists at {EXTRACT_DIR}, skipping download.")
        return

    print(f"Downloading MovieLens 1M from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print("Download complete.")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {EXTRACT_DIR}")

    os.remove(ZIP_PATH)
    print("Cleaned up zip file.")

    # Verify expected files exist
    expected = ["ratings.dat", "movies.dat", "users.dat"]
    for fname in expected:
        fpath = os.path.join(EXTRACT_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected file not found: {fpath}")
        line_count = sum(1 for _ in open(fpath, encoding="latin-1"))
        print(f"  {fname}: {line_count} lines")

    print("MovieLens 1M dataset ready.")


if __name__ == "__main__":
    download_and_extract()

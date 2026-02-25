import requests
import os
from zipfile import ZipFile

for n in range(1, 14):
    full_url = f"https://download.open.fda.gov/drug/label/drug-label-00{str(n).zfill(2)}-of-0013.json.zip"
    file_name = os.path.basename(full_url)

    print(f"Downloading {file_name}...")

    file_response = requests.get(full_url, stream=True)
    file_response.raise_for_status()

    with open(file_name, "wb") as f:
        for chunk in file_response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    try:
        with ZipFile(file_name, "r") as zip_ref:
            zip_ref.extractall("./")
        print(f"Successfully unzipped {file_name}")
    except Exception as e:
        print(f"Error unzipping file: {e}")

    print(f"Downloaded {file_name}")
    os.remove(file_name)

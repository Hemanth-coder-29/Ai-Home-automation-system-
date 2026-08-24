import os
import shutil
import requests
import zipfile
from tqdm import tqdm

MODEL_DIR = "model"
ZIP_PATH = os.path.join(MODEL_DIR, "vosk_model.zip")
EXTRACTED_DIR = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")
FINAL_DIR = os.path.join(MODEL_DIR, "vosk-model")
URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Clean up broken files from previous attempt
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)
if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)
if os.path.exists(EXTRACTED_DIR):
    shutil.rmtree(EXTRACTED_DIR)

# Download with a progress bar
print("Downloading Vosk offline voice model (40 MB)...")
response = requests.get(URL, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(ZIP_PATH, 'wb') as file, tqdm(
    desc="Progress",
    total=total_size,
    unit='iB',
    unit_scale=True
) as bar:
    for data in response.iter_content(chunk_size=1024):
        size = file.write(data)
        bar.update(size)

# Extract
print("\nExtracting files...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(MODEL_DIR)

# Rename and cleanup
os.rename(EXTRACTED_DIR, FINAL_DIR)
os.remove(ZIP_PATH)

print("Vosk model successfully downloaded and ready for use!")
import urllib.request
import zipfile
import io
import os

# Create the target directory if it doesn't exist
DATA_DIR = 'nltk_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the URLs for the NLTK data packages
urls = {
    'punkt': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip',
    'stopwords': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip'
}

# Download and unzip each package
for name, url in urls.items():
    print(f"Downloading and unzipping {name}...")
    try:
        with urllib.request.urlopen(url) as response:
            # Read the response into a byte stream
            zip_content = io.BytesIO(response.read())
            # Create a ZipFile object and extract all its contents
            with zipfile.ZipFile(zip_content) as zf:
                zf.extractall(DATA_DIR)
        print(f"{name} downloaded successfully.")
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        exit(1) # Exit with an error code if download fails

print("NLTK setup completed successfully.")

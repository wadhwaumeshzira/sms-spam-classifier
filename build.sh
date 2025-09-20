#!/bin/bash

# Exit on any error
set -o errexit

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Create a directory for NLTK data within the project
mkdir -p nltk_data

# Run the NLTK downloader and specify the target directory
python -c "import nltk; nltk.download('punkt', download_dir='nltk_data'); nltk.download('stopwords', download_dir='nltk_data')"


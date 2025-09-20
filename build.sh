#!/bin/bash

# Exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Run the NLTK downloader
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

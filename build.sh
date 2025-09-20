#!/bin/bash

# Exit on any error
set -o errexit

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# --- Create directories for NLTK data ---
# This ensures the folder structure is correct before unzipping
mkdir -p nltk_data/tokenizers
mkdir -p nltk_data/corpora

# --- Download NLTK data directly from the official source ---
echo "Downloading NLTK 'punkt' tokenizer..."
wget -O punkt.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip

echo "Downloading NLTK 'stopwords' corpora..."
wget -O stopwords.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip

# --- Unzip the data into the correct subdirectories ---
echo "Unzipping NLTK data..."
unzip -o punkt.zip -d nltk_data/tokenizers
unzip -o stopwords.zip -d nltk_data/corpora

# --- Clean up the downloaded zip files ---
echo "Cleaning up..."
rm punkt.zip
rm stopwords.zip

echo "Build script finished successfully."


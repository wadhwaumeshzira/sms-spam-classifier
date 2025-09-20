#!/bin/bash

# Exit on any error
set -o errexit

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Run our Python script to download and set up NLTK data
python setup_nltk.py


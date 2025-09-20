import string
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import urllib.request
import zipfile
import io

# --- NLTK Data Setup ---
# This block of code ensures that the necessary NLTK data is available.
# It checks for the data and downloads it if it's missing.

DATA_DIR = 'nltk_data'
REQUIRED_PACKAGES = {
    'tokenizers/punkt': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip',
    'corpora/stopwords': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip'
}

def setup_nltk_data():
    """
    Checks for NLTK data and downloads it if missing.
    """
    print("Checking for NLTK data...")
    # Add our local directory to NLTK's data path
    nltk.data.path.append(os.path.abspath(DATA_DIR))
    
    # Create the main nltk_data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for path, url in REQUIRED_PACKAGES.items():
        full_path = os.path.join(DATA_DIR, path)
        if not os.path.exists(full_path):
            print(f"Data package '{os.path.basename(path)}' not found. Downloading...")
            try:
                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with urllib.request.urlopen(url) as response:
                    zip_content = io.BytesIO(response.read())
                with zipfile.ZipFile(zip_content) as zf:
                    zf.extractall(os.path.join(DATA_DIR, os.path.dirname(path)))
                print(f"Successfully downloaded and unzipped '{os.path.basename(path)}'.")
            except Exception as e:
                print(f"FATAL: Failed to download NLTK data package '{os.path.basename(path)}'. Error: {e}")
                exit(1) # Stop the application if data can't be downloaded
        else:
            print(f"Data package '{os.path.basename(path)}' already exists.")

# Run the setup function when the application starts
setup_nltk_data()


# --- Load Model and Vectorizer ---
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("FATAL: 'vectorizer.pkl' or 'model.pkl' not found.")
    exit()

# --- Text Preprocessing Function ---
ps = nltk.stem.porter.PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    y = [i for i in y if i not in stop_words and i not in punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)


# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "SMS Spam Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.json['message']
        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        prediction = "Spam" if result == 1 else "Not Spam (Ham)"
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)


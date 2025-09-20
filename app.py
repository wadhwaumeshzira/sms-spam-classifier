from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --- Download NLTK Data ---
# This code will now run automatically when the app starts on Render.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpus/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


# --- Text Preprocessing Function ---
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Lowercase
    text = nltk.word_tokenize(text)  # Tokenize

    # Remove special characters and retain alphanumeric words
    y = [i for i in text if i.isalnum()]

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    y = [i for i in y if i not in stop_words and i not in punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)


# --- Load Model and Vectorizer ---
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    print("Please make sure these files are in the same directory as app.py")
    # This will cause the app to fail on startup if models are missing
    raise


# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

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

        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam (Ham)"

        return jsonify({'prediction': prediction})

    except Exception as e:
        print(e)
        return jsonify({'error': 'An error occurred during prediction.'}), 500


# The if __name__ == '__main__' block is only for local development
# and is not needed for the Render deployment.
if __name__ == '__main__':
    app.run(debug=True)


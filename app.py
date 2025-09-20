from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --- Text Preprocessing Function ---
# This should be the SAME function you used to train your model.
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
# Make sure these files are in the same directory as app.py
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    print("Please make sure these files are in the same directory as app.py")
    exit() # Exit if model files are not found


# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

@app.route('/')
def home():
    return "SMS Spam Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the message from the POST request
        message = request.json['message']

        # 1. Preprocess the message
        transformed_sms = transform_text(message)

        # 2. Vectorize the message
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict using the model
        result = model.predict(vector_input)[0]

        # 4. Return the result as JSON
        # 1 means Spam, 0 means Not Spam (Ham)
        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam (Ham)"

        return jsonify({'prediction': prediction})

    except Exception as e:
        print(e)
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == '__main__':
    # Download necessary NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpus/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
        
    app.run(debug=True)
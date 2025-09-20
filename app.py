from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# NOTE: All nltk.download() calls have been removed from this file.
# The build.sh script now handles the download process.

# --- Text Preprocessing Function ---
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    y = [i for i in y if i not in stop_words and i not in punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# --- Load Model and Vectorizer ---
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' or 'model.pkl' not found.")
    raise

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
        print(e)
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)


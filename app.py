from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Ensure NLTK resources are available ---
# Download only what is required
nltk.download('punkt')
nltk.download('stopwords')

# Optional: if you bundle nltk_data locally, keep this line
nltk.data.path.append('./nltk_data')

# --- Text Preprocessing Function ---
ps = PorterStemmer()

def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stopwords & punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    # Stemming
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)

# --- Load Model and Vectorizer ---
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    raise RuntimeError("Error: 'vectorizer.pkl' or 'model.pkl' not found in project root.")

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "✅ SMS Spam Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" in request body'}), 400

        message = data['message']
        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        prediction = "Spam" if result == 1 else "Not Spam (Ham)"
        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    # Run on all interfaces (useful for Render/local Docker)
    app.run(host="0.0.0.0", port=5000, debug=True)

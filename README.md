# SMS Spam Classifier



🚀 **Smart SMS Spam Detector – Machine Learning & NLP for clean messaging**

A Machine Learning and NLP-based SMS Spam Classifier that detects whether an SMS message is **Spam** or **Ham (not spam)**. The project uses text preprocessing, feature extraction, and ML modeling to achieve strong performance and real-world usability.

---

## 📚 Project Files

| File | Description |
|---|---|
| `sms spam classifier.ipynb` | Jupyter Notebook with end-to-end pipeline: data cleaning, feature engineering, model training and evaluation. |
| `spam[1].csv` | The dataset containing labeled SMS messages (spam / ham). |
| `model.pkl` | Saved trained ML model (pickled) for inference. |
| `vectorizer.pkl` | Saved vectorizer (e.g. TF-IDF or Bag-of-Words) used to convert text to features. |
| `README.md` | Project description, usage, etc. |
| `LICENSE` | MIT License for open source reuse. |

---

## ⚙️ Features & Pipeline

1. **Text Preprocessing**  
   - Cleaning raw SMS data (lowercasing, removing punctuation)  
   - Tokenization, stopword removal, optional stemming or lemmatization  

2. **Feature Extraction**  
   - Bag-of-Words or TF-IDF vectorization from text data via the `vectorizer.pkl`  

3. **Modeling**  
   - Training of machine learning classifiers (e.g. Naive Bayes, Logistic Regression, etc.)  
   - Saved model is in `model.pkl`  

4. **Evaluation**  
   - Metrics like accuracy, precision, recall, F1-score  
   - Confusion matrix, ROC/AUC may be used  

5. **Usage**  
   - Load vectorizer & model to predict new SMS messages  
   - Can be extended into a web app or API  

---

## 📊 Results & Performance

- High classification accuracy (often > 95%) in distinguishing spam vs ham  
- Good precision & recall on spam class to reduce false positives/negatives  
- Model generalizes well on test samples  

*(You can add your actual metric values here: Accuracy, F1-Score, etc.)*

---

## 🚀 How to Use / Run

1. Clone the repository:

   ```bash
   git clone https://github.com/wadhwaumeshzira/sms-spam-classifier.git
   cd sms-spam-classifier

2. **Install dependencies**

Make sure you have Python installed. Then run:

```pip install -r requirements.txt```


3. **Train / Evaluate or Load existing model**

- Open the notebook `sms spam classifier.ipynb` to walk through training and evaluation.
- To use the pretrained model for prediction:

```python
import pickle

# Load vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Example prediction
message = "Congratulations! You've won a free ticket."
features = vectorizer.transform([message])
prediction = model.predict(features)
print("Prediction:", prediction[0])

| SMS Message                                                    | Prediction |
| -------------------------------------------------------------- | ---------- |
| “Congratulations! You’ve won a free lottery ticket. Call now.” | **Spam**   |
| “Hey, are we meeting at 7 pm today?”                           | **Ham**    |

---
```
## 👤 Developer

**Umesh Kumar**  

- B.Tech in Information Technology (2023–2027) at IIIT Bhopal  
- Email: umeshwadhwa11@gmail.com
- Phone: +91 7719404884
- GitHub: [https://github.com/wadhwaumeshzira](https://github.com/wadhwaumeshzira)  

> This project is developed as part of my hands-on experience in machine learning and natural language processing projects.  
> It demonstrates building a practical SMS spam classifier using Python, scikit-learn, and NLP techniques.

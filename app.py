import spacy
import nltk
import pandas as pd
import re
import joblib
import math
import os
import string
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from transformers import TFRobertaForSequenceClassification, pipeline, RobertaTokenizerFast

# Flask app initialization
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

# Setting up paths
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'Saved_Models', 'updated_predictor.pkl')
vectorizer_path = os.path.join(base_path, 'Saved_Models', 'updated_vectorizer.pkl')
emo_model_path = os.path.join(base_path, 'Saved_Models', 'EmoRoBERTa_model')
tokenizer_path = os.path.join(base_path, 'Saved_Models', 'EmoRoBERTa_tokenizer')

# Loading the Spam Detection model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Loading the Emotional Analysis model and tokenizer
emo_model = TFRobertaForSequenceClassification.from_pretrained(emo_model_path)
emo_tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
emotion = pipeline('sentiment-analysis', model=emo_model, tokenizer=emo_tokenizer, return_all_scores=True)

# Initializing NLTK libraries
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to remove html links
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script', 'code', 'a']):
        data.decompose()
    return ' '.join(soup.stripped_strings)

# Function to clean and lemmatize the string
def clean_string(text, stem='Spacy'):
    lemmatizer = WordNetLemmatizer()
    clean_text = re.sub(r'\s+', ' ', text)
    clean_text = ' '.join([lemmatizer.lemmatize(word) for word in clean_text.split() if word not in stopwords.words('english')])
    return clean_text

# Function to sort out Emotional Labels
def sort_emo(result):
    sorted_emotions = sorted(result[0], key=lambda x: x['score'], reverse=True)
    top_5_emotions = sorted_emotions[:5]
    return top_5_emotions

# Route for the main page
@app.route('/')
def page():
    return render_template('test.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        if not email:
            return render_template('test.html', error_message='Please enter an email.')
        else:
            count_of_words = len(email.split())

            # Perform Emotional Analysis
            result = emotion(email)  # Emotional Analysis Labels
            sorted_labels = sort_emo(result)  # Sorting in Descending order; extracts top 5 labels

            # Clean the text
            clean_text = clean_html(email)
            processed_text = clean_string(clean_text)

            # Perform Spam Detection prediction
            string_vectorized = vectorizer.transform([processed_text])
            my_prediction = model.predict(string_vectorized)
            probability = model.predict_proba(string_vectorized)[0][1]
            percentage = round(probability * 100, 2)

            # Capitalize emotion labels
            capitalized_labels = []
            scores = []

            # For extracting top 5 labels
            for res in sorted_labels:
                label = res['label']
                capitalized_label = label.capitalize()
                score = round(res['score'] * 100, 2)
                capitalized_labels.append(capitalized_label)
                scores.append(score)

            # Zipping the lists together
            emo_data = zip(capitalized_labels, scores)

        return render_template('test.html', prediction=my_prediction, percentage=percentage, result=count_of_words, email=email, emo_data = emo_data)  # Pass the email value back to the template

    # For GET request or if no form submission has occurred
    return render_template('test.html', email='')  # Pass an empty string as the email value

if __name__ == '__main__':
    app.run(debug=True)
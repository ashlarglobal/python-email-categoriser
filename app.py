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

#================================================================================================================================#

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

#================================================================================================================================#

# Dictionary to map emotions to CSS color values
emotion_colors = {
    'admiration': '#BD62FF',
    'amusement': '#FFF17B',
    'anger': '#FF6B6B',
    'annoyance': '#FFA954',
    'approval': '#7DFF7D',
    'caring': '#FFD4E8',
    'confusion': '#E0E0E0',
    'curiosity': '#A2CDFF',
    'desire': '#FF6F85',
    'disappointment': '#BEBEBE',
    'disapproval': '#724BA3',
    'disgust': '#B8C375',
    'embarrassment': '#FF9CC6',
    'excitement': '#FFC65C',
    'fear': '#936DBF',
    'gratitude': '#FFEE8C',
    'grief': '#3B40A3',
    'joy': '#FFF960',
    'love': '#FF7878',
    'nervousness': '#CDDFF0',
    'optimism': '#B8E2FA',
    'pride': '#FFEE8C',
    'realization': '#FFFFB3',
    'relief': '#8FFF8F',
    'remorse': '#5DA5A5',
    'sadness': '#BEBEBE',
    'surprise': '#FF9AFF'
}

#================================================================================================================================#

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

# Function to process text emotions
def process_text_emotions(processed_text):
    temp_list = processed_text.split()
    emotion_lists = {
        'admiration': [], 'amusement': [], 'anger': [], 'annoyance': [],
        'approval': [], 'caring': [], 'confusion': [], 'curiosity': [],
        'desire': [], 'disappointment': [], 'disapproval': [], 'disgust': [],
        'embarrassment': [], 'excitement': [], 'fear': [], 'gratitude': [],
        'grief': [], 'joy': [], 'love': [], 'nervousness': [], 'optimism': [],
        'pride': [], 'realization': [], 'relief': [], 'remorse': [], 'sadness': [],
        'surprise': []
    }

    for x in temp_list:
        results = emotion(x)
        if results != 'neutral':
            for result in results:
                label_max = max(result, key=lambda dictionary: dictionary['score'])
                label = label_max["label"]
                if label in emotion_lists:
                    emotion_lists[label].append(x)

    # Removing empty lists from emotion_lists
    emotion_lists = {key: value for key, value in emotion_lists.items() if value}
    return emotion_lists

# Function to generate the highlighted HTML text
def generate_html_with_highlights(text, emotions):
    # Split text into words and punctuation marks
    word_pattern = re.compile(r'\w+|[^\w\s]')
    tokens = word_pattern.findall(text)

    highlighted_tokens = []
    for i, token in enumerate(tokens):
        if token.isalpha():
            for emotion, word_list in emotions.items():
                if token.lower() in word_list:
                    highlighted_tokens.append(f'<mark style="background-color:{emotion_colors[emotion]}">{token}</mark>')
                    break
            else:
                highlighted_tokens.append(token)
        else:
            highlighted_tokens.append(token)

        # Add space after the token if it's not the last token and the next token is not a punctuation mark
        if i < len(tokens) - 1 and not re.match(r'[^\w\s]', tokens[i+1]):
            highlighted_tokens.append(' ')

    highlighted_text = ''.join(highlighted_tokens)
    return highlighted_text

#================================================================================================================================#

# Route for the main page
@app.route('/')
def page():
    return render_template('testing.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        if not email:
            return render_template('testing.html', error_message='Please enter an email.')
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
            
            # Process text emotions
            emotion_lists = process_text_emotions(processed_text)

            # Generate highlighted HTML text
            highlighted_text = generate_html_with_highlights(email, emotion_lists)

            return render_template('testing.html', prediction=my_prediction, percentage=percentage, result=count_of_words, email=email, emo_data=emo_data, highlighted_text=highlighted_text)  # Pass the email value back to the template

    # For GET request or if no form submission has occurred
    return render_template('testing.html', email='')  # Pass an empty string as the email value

if __name__ == '__main__':
    app.run(debug=True)

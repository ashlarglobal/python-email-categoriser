import spacy
import nltk
import pandas as pd
import re
import joblib
import math
import os
import string
import warnings
import json
warnings.filterwarnings("ignore")

from flask import Flask, render_template, url_for, request, jsonify
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
# Declaring Global Variables

prediction = None
percentage = None
result = None
email = None
emo_data = None
highlighted_text = None

#================================================================================================================================#
# Dictionary to map emotions to CSS color values
emotion_colors = {
    'affectionate': '#FF9AFF',
    'energetic': '#CCFF99',
    'contentment': '#FFA954',
    'irritated': '#FF7878',
    'sorrowful': '#A2CDFF',
    'inquisitive': '#7DFF7D',
    'reflective': '#5DA5A5',
    'apprehensive': '#724BA3'
}

#================================================================================================================================#

# Function to remove html links
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script', 'code', 'a']):
        data.decompose()
    return ' '.join(soup.stripped_strings)

# For truncating the tokens
def truncate_text(text, max_tokens):
    tokenizer = pipeline('sentiment-analysis').tokenizer
    encoded_text = tokenizer(text, truncation=True, max_length=max_tokens, padding='longest')
    truncated_text = tokenizer.decode(encoded_text['input_ids'][0], skip_special_tokens=True)
    return truncated_text

# Function to clean and lemmatize the string with token limit
def clean_string(text, max_tokens, stem='Spacy'):
    lemmatizer = WordNetLemmatizer()
    clean_text = re.sub(r'\s+', ' ', text)
    tokenized_text = word_tokenize(clean_text)
    truncated_tokens = tokenized_text[:max_tokens]
    truncated_text = ' '.join([lemmatizer.lemmatize(word) for word in truncated_tokens if word not in stop_words])
    return truncated_text

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

# Function for Grouping emotions
def create_reduced_emotions(emotion_lists):
    reduced_emotions = {
        'affectionate': [],
        'energetic': [],
        'contentment': [],
        'irritated': [],
        'sorrowful': [],
        'inquisitive': [],
        'reflective': [],
        'apprehensive': []
    }

    for emotion, words in emotion_lists.items():
        if emotion in ['love', 'admiration', 'caring', 'gratitude', 'desire', 'pride']:
            reduced_emotions['affectionate'].extend(words)
        elif emotion in ['joy', 'excitement', 'amusement', 'optimism', 'surprise']:
            reduced_emotions['energetic'].extend(words)
        elif emotion in ['approval', 'relief']:
            reduced_emotions['contentment'].extend(words)
        elif emotion in ['anger', 'disapproval', 'annoyance', 'disgust']:
            reduced_emotions['irritated'].extend(words)
        elif emotion in ['grief', 'sadness', 'remorse']:
            reduced_emotions['sorrowful'].extend(words)
        elif emotion in ['curiosity', 'confusion']:
            reduced_emotions['inquisitive'].extend(words)
        elif emotion in ['realization', 'disappointment', 'embarrassment']:
            reduced_emotions['reflective'].extend(words)
        elif emotion in ['fear', 'nervousness']:
            reduced_emotions['apprehensive'].extend(words)
            
    return reduced_emotions

# Function to generate the highlighted HTML text
def generate_html_with_highlights(text, emotions):
    # Split text into words and punctuation marks
    word_pattern = re.compile(r'\w+|[^\w\s]')
    tokens = word_pattern.findall(text)

    highlighted_tokens = []
    for i, token in enumerate(tokens):
        if any(token.lower() in word_list or token.capitalize() in word_list or token.upper() in word_list for word_list in emotions.values()):
            for emotion, word_list in emotions.items():
                if token.lower() in word_list or token.capitalize() in word_list or token.upper() in word_list:
                    highlighted_tokens.append(f'<mark style="background-color:{emotion_colors[emotion]}">{token}</mark>')
                    break
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

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.json['email']
        if not email:
            return json.dumps({'error': 'Please enter an email.'})
        else:
            count_of_words = len(email.split())

            # Truncate email text to 512 tokens
            truncated_text = clean_string(email, max_tokens=512)
            
            # Perform Emotional Analysis
            result = emotion(truncated_text)  # Emotional Analysis Labels
            sorted_labels = sort_emo(result)  # Sorting in Descending order; extracts top 5 labels

            # Clean the text
            clean_text = clean_html(email)
            processed_text = clean_string(clean_text, max_tokens=512)

            # Perform Spam Detection prediction
            string_vectorized = vectorizer.transform([processed_text])
            my_prediction = model.predict(string_vectorized)
            my_prediction = my_prediction.tolist()  # Convert ndarray to list
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

            # Return the result box HTML content as JSON
            return json.dumps({
                'prediction': my_prediction,
                'percentage': percentage,
                'result': count_of_words,
                'emo_data': list(emo_data)
            })

@app.route('/highlight_text', methods=['POST'])
def highlight_text():
    if request.method == 'POST':
        email = request.json['email']
        
        # Perform necessary processing to generate highlighted text
        processed_text = clean_string(clean_html(email), max_tokens=512)
        emotion_lists = process_text_emotions(processed_text)
        reduced_emotions = create_reduced_emotions(emotion_lists)
        highlighted_text = generate_html_with_highlights(email, reduced_emotions)
        
        return highlighted_text

# Handle cases where the request method is not POST 
if __name__ == '__main__': 
	app.run(debug=True)
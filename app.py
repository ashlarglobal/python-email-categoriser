import pandas as pd
import re # Regex functions
import joblib
import math
import os

# Flask libraries
from flask import Flask, render_template,url_for,request

# Model Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Text preprocessing libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to remove stopwords from text
def rmv_stopwords(string):
    tokens = word_tokenize(string)
    filtered_tokens = [w for w in tokens if not w in stop_words]
    filtered_sentence = ' '.join(filtered_tokens)
    return filtered_sentence

# Function to Lemmatize the words in a text
def lemmatize_words(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return ' '.join(words)

# Function to Preprocess the dataframe
def pre_proc(data):
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", '', data)
    text = re.sub(r'\b\w{10,}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('textplain', '')
    return text

app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

# Load the pre-trained model and TF-IDF vectorizer
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'Saved_Models', 'spam_predictor.pkl')
vectorizer_path = os.path.join(base_path, 'Saved_Models', 'tfidf_vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def page():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        if not email:
            return render_template('test.html', error_message='Please enter an email.')
        else:
            count_of_words = len(email.split()) # Counting number of words
            
            clean_text = pre_proc(email) # Pre-processing of text
            processed_text = lemmatize_words(rmv_stopwords(clean_text))
            string_vectorized = vectorizer.transform([processed_text])
            my_prediction = model.predict(string_vectorized)
            
            probability = model.predict_proba(string_vectorized)[0][1]  # Probability of being spam
            percentage = round(probability * 100, 2) # Percentage of being spam
            
    return render_template('test.html', prediction=my_prediction,  percentage=percentage, result=count_of_words)

if __name__ == '__main__':
    app.run(debug=True)
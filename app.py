import pandas as pd
import re # Regex functions
import joblib
import os

# Flask libraries
from flask import Flask, render_template,url_for,request

# Model Libraries
from sklearn.feature_extraction.text import TfidfVectorizer

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

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model and TF-IDF vectorizer
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'Saved_Models', 'spam_predictor.pkl')
vectorizer_path = os.path.join(base_path, 'Saved_Models', 'tfidf_vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def page():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        if not email:
            return render_template('page.html', error_message='Please enter an email.')
        else:
            clean_text = pre_proc(email)
            processed_text = lemmatize_words(rmv_stopwords(clean_text))
            string_vectorized = vectorizer.transform([processed_text])
            my_prediction = model.predict(string_vectorized)
    return render_template('page.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
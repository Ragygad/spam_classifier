# Import required dependencies
from flask import Flask,render_template,url_for,request
import pickle
import re
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import collections
import nltk
import os
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB




vect = pickle.load(open('vectorizer_final.plk','rb'))
model = pickle.load(open('model_Classifier.plk','rb'))

app = Flask(__name__)

# Preprocessing function: new_twt is user input, model/cv are created/available above
def classify_email_message(input1, model1,cv):  
    model = model1
    vect = cv

    data = [input1]
    vect = cv.transform(data).toarray()
    my_prediction = model.predict(vect)
    
    return my_prediction[0]

#
# Preprocessing Functions end
@app.route("/", methods=["GET"])
def index():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form['message']
    answer = classify_email_message(message,model,vect)
		
    return render_template('results.html',prediction = answer, message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000, debug=True)




# dependencies
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Loading Joblib 
multinomialNbModel = joblib.load('./Model/multinomialNB.pkl') 
preprocessor = joblib.load('./Model/preprocessor.pkl')
encoder_category = joblib.load('./Model/encoder_category.pkl')

# For Lemmatizing
word_lama = WordNetLemmatizer()
# stopwords list
stopwords_list = stopwords.words("english")

# Cleaning for input data
def text_cleaning(text):
    # Remove the dot from floating-point numbers and replace non-alphanumeric characters with a space
    cleaned_text = re.sub(r"\b(\d+)\.\d+\b",r'\1', text)
    cleaned_text = re.sub(r"[^a-zA-Z0-9]", " ", cleaned_text.lower())
    # creating tokens
    cleaned_text = word_tokenize(cleaned_text,language="english")
    # lemmatizing text
    lemmatizing_data = [word_lama.lemmatize(word) for word in cleaned_text]
    # stop words removal
    cleaned_text = [' '.join([word for word in lemmatizing_data if not word in stopwords_list])]
    return cleaned_text[0]


app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return jsonify({'MESSAGE':'WebNews API for Classification of News according to Genre.','status':200})

@app.route('/classify',methods=['POST'])
def classify():
    data = request.get_json()
    news_headline = data['news_headline']
    news_article = data['news_article']
    
    # Predicting Data
    data_test = pd.DataFrame({'news_headline': [news_headline] ,'news_article':[news_headline]})

    # preprocessing input data
    data_test['news_headline'] = data_test['news_headline'].apply(lambda headline: text_cleaning(headline))
    data_test['news_article'] = data_test['news_article'].apply(lambda article: text_cleaning(article))
    data_test = preprocessor.transform(data_test)
    
    # Decode the encoded labels
    news_genre = encoder_category.inverse_transform(multinomialNbModel.predict(data_test))
    # print("Prediction of a new datapoint : ",news_genre[0])
    
    # Returning result
    return jsonify({'News Headline':news_headline, 'News Article': news_article, 'Genre': news_genre[0] ,'status':200})

if __name__ == '__main__':
    app.run(debug=True)
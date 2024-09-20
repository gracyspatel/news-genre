# importing dependencies
import re
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Loading Joblib 
multinomialNbModel = joblib.load('./Model/multinomialNB.pkl') 
preprocessor = joblib.load('./Model/preprocessor.pkl')
encoder_category = joblib.load('./Model/encoder_category.pkl')

# Example Data
new_headline="Shiv Sena, BJP to contest all upcoming polls together: Maha CM"
new_article = """Maharashtra CM Eknath Shinde has announced that the Shiv Sena and BJP will contest all the upcoming elections in the state, including Lok Sabha, Vidhan Sabha and local body elections, together. This comes after Shinde and state Deputy CM Devendra Fadnavis met Union Home Minister Amit Shah in Delhi. He stated that their alliance is "strong" for the state's development."""

# Predicting Data
data_test = pd.DataFrame({'news_headline': [new_headline] ,'news_article':[new_article]})

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


# special symbols removal
data_test['news_headline'] = data_test['news_headline'].apply(lambda headline: text_cleaning(headline))
data_test['news_article'] = data_test['news_article'].apply(lambda article: text_cleaning(article))
data_test = preprocessor.transform(data_test)
# Decode the encoded labels
news_genre = encoder_category.inverse_transform(multinomialNbModel.predict(data_test))
print("Prediction of a new datapoint : ",news_genre[0])

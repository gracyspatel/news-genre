# importing dependencies
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
import joblib

# reading file
data = pd.read_csv("./Data/inshort_news_data-1.csv")

for i in range(2,7):
    data = pd.concat([data,pd.read_csv("./Data/inshort_news_data-"+str(i)+".csv")])

# dropping Unnamed:0 column
data.drop(columns=['Unnamed: 0'],axis=1,inplace=True)

# Dealing with duplicated values (removing duplicate values)
data = data.drop_duplicates()

# Label Encoding target column
encoder_category = LabelEncoder()
data['news_category'] = encoder_category.fit_transform(data['news_category'])

# target feature splitting
feature = data.iloc[:,:-1]
target = data.iloc[:,-1]

# Headline cleaning and processing
# For Lemmatizing
word_lama = WordNetLemmatizer()
stopwords_list = stopwords.words("english")

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
feature['news_headline'] = feature['news_headline'].apply(lambda headline: text_cleaning(headline))
feature['news_article'] = feature['news_article'].apply(lambda headline: text_cleaning(headline))

# train test split
x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size=0.20)

# Applying Count Vectorize
# (Bag of Words)
vectorizer = CountVectorizer()

# Preprocess the text columns using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('headline', vectorizer, 'news_headline'),
        ('article', vectorizer, 'news_article')
    ])

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

smote_sampling = SMOTE()
x_train, y_train = smote_sampling.fit_resample(x_train,y_train)

mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_predicted = mnb.predict(x_test)

# Save the model
joblib.dump(mnb, './Model/multinomialNB.pkl') 
# Saving necessary objects
joblib.dump(preprocessor, './Model/preprocessor.pkl')
joblib.dump(encoder_category,'./Model/encoder_category.pkl')

# Model Generated
print("Model Joblib File Successfully saved")
# importing dependencies
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE

# nltk
import nltk
# Downloading stopwords
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# Stopwords List
stopwords_list = stopwords.words("english")
print("\nStop-words : ")
print(stopwords_list)
print("Total Stop Words in corpus : ",len(stopwords_list))

# reading csv
data = pd.read_csv("./Data/inshort_news_data-1.csv")

for i in range(2,7):
    data = pd.concat([data,pd.read_csv("./Data/inshort_news_data-"+str(i)+".csv")])

# printing data
print("Data :")
print(data.head())

# shape
print("Shape : ",data.shape)

# dropping Unnamed:0 column
data.drop(columns=['Unnamed: 0'],axis=1,inplace=True)

# Columns
print("Column Names : ", data.columns.values)

# Unique Target columns
print(data['news_category'].unique())

# Value counts
print(data['news_category'].value_counts())

# Checking null values
print("\nNull Values : ")
print(data.isnull().sum())

# Checking duplicated values
print("\nDuplicated Values : ")
print(data.duplicated().sum())

# Dealing with duplicated values
data = data.drop_duplicates()

# Checking duplicated values
print("\nDuplicated Values : ")
print(data.duplicated().sum())

# Value counts
print(data['news_category'].value_counts())
print("\nIndex : ",data['news_category'].value_counts().index.tolist())
print("Values : ",data['news_category'].value_counts().values.tolist())

# Count plot
plt.figure(figsize=(10,5))
sns.countplot(data=data,x='news_category')
plt.show()

# Box plot for the genre count
plt.figure(figsize=(4,4))
plt.title("Box Plot")
sns.boxplot(data=data['news_category'].value_counts().values.tolist())
plt.show()

# Histogram
plt.figure(figsize=(10,7))
labels = data['news_category'].value_counts().index.tolist()
counts = data['news_category'].value_counts().values.tolist()
plt.bar(labels, counts)
plt.xlabel('Class Labels')
plt.ylabel('Count')
plt.title('Histogram of Target Column')
plt.show()

# Label Encoding target column
encoder_category = LabelEncoder()
data['news_category'] = encoder_category.fit_transform(data['news_category'])

# target feature splitting
feature = data.iloc[:,:-1]
target = data.iloc[:,-1]

# Headline cleaning and processing

# printing data
print("Feature:")
print(feature.head())

# For Lemmatizing
word_lama = WordNetLemmatizer()

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

svc = MultinomialNB()
svc.fit(x_train,y_train)
y_predicted = svc.predict(x_test)

# Accuracy Sore
print(accuracy_score(y_true=y_test,y_pred=y_predicted)*100)
print(confusion_matrix(y_true=y_test,y_pred=y_predicted))
print(classification_report(y_true=y_test,y_pred=y_predicted))

# Example Data
new_headline="Shiv Sena, BJP to contest all upcoming polls together: Maha CM"
new_article = """Maharashtra CM Eknath Shinde has announced that the Shiv Sena and BJP will contest all the upcoming elections in the state, including Lok Sabha, Vidhan Sabha and local body elections, together. This comes after Shinde and state Deputy CM Devendra Fadnavis met Union Home Minister Amit Shah in Delhi. He stated that their alliance is "strong" for the state's development."""

data_test = pd.DataFrame({'news_headline': [new_headline] ,'news_article':[new_article]})
# special symbols removal
data_test['news_headline'] = data_test['news_headline'].apply(lambda headline: text_cleaning(headline))
data_test['news_article'] = data_test['news_article'].apply(lambda article: text_cleaning(article))
data_test = preprocessor.transform(data_test)
# Decode the encoded labels
news_genre = encoder_category.inverse_transform(svc.predict(data_test))
print("Prediction of a new datapoint : ",news_genre[0])
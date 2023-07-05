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
nltk.download('stopwords')
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

new_headline = "Elon Musk: Twitter snaps up top NBCUniversal executive"
new_article= "Twitter has snapped up senior NBCUniversal executive Joe Benarroch, as Elon Musk " \
          "continues to shake up the social media platform's top team. Mr Benarroch starts the " \
          "role on Monday, focussing on business operations. It comes less than a month after NBCUniversal's head of advertising Linda Yaccarino was named as Twitter's new chief executive." \
          "She will take the position currently held by Mr Musk, who will remain closely involved in the company."

# new_headline = "Twitch streamer Puppers, who lived with MND, dies aged 32"
# new_article= "Popular streamer Puppers has died aged 32, three years after being diagnosed with motor neurone disease (MND).The US Twitch streamer rose to fame playing survival game Dead by Daylight, where he was known for his positivity.The game's community rallied around him following his diagnosis and set up the Light in the Fog Foundation, raising $270k (£216k) to support his care.In a reference to his catchphrase, the foundation said Puppers was forever in our hearts, eternally comfy.Puppers, also known as Max, would end his streams by telling fans to stay comfy, because if you're comfy, you're winning.The streamer was diagnosed with ALS - the most common form of MND - in 2020.It affects the brain and nerves and causes weakness that gets worse over time, according to the NHS. There is no cure for the disease, but there are treatments to help reduce the impact on a person's daily life.A post on his Twitter page confirmed his death, saying he loved you all so very much.Thank you for all of the love and support throughout his career - making you all happy is truly what he lived for, it said."

# new_headline = "iPhone in India: Foxconn to manufacture smartphones in Karnataka by April 2024"
# new_article= """Apple's biggest supplier Foxconn will start manufacturing iPhones in the southern Indian state of Karnataka by April next year, the state government has said. The project will create around 50,000 jobs, it said.Taiwan-based Foxconn manufactures the majority of Apple's phones.The firm has been making older versions of iPhones at a facility in the neighbouring state of Tamil Nadu since 2017.Last month, the company announced it had bought 1.2m sqm (13m sqft) of land near Bengaluru city in Karnataka.Bloomberg reported Foxconn planned to invest $700m (£566m) on a new factory in the state. On Thursday, the Karnataka government said the project was valued at $1.59bn.Land for the factory would be handed over to company by 1 July, it said in its statement.According to Reuters, Foxconn has set a target of manufacturing 20 million iPhones a year at the plant in Karnataka."""


# new_headline = "Powerful artificial intelligence ban possible, government adviser warns"
# new_article = """Some powerful artificial general intelligence AGI systems may eventually have to
# be banned, a member of the governments AI Council says.Marc Warner, also boss of Faculty AI, told the BBC that AGI needed strong transparency and audit requirements as well as more inbuilt safety technology.And the next six months to a year would require sensible decisions on AGI.His comments follow the EU and US jointly saying a voluntary code of practice for AI was needed soon."""

# political news
# new_headline="Shiv Sena, BJP to contest all upcoming polls together: Maha CM"
# new_article = """Maharashtra CM Eknath Shinde has announced that the Shiv Sena and BJP will contest all the upcoming elections in the state, including Lok Sabha, Vidhan Sabha and local body elections, together. This comes after Shinde and state Deputy CM Devendra Fadnavis met Union Home Minister Amit Shah in Delhi. He stated that their alliance is "strong" for the state's development."""

# new_headline="Bhagalpur bridge keeps falling due to faulty construction: Nitish"
# new_article = """Bhagalpur bridge keeps falling due to faulty construction: Nitish
# short by Subhangi Singh / 12:10 pm on 05 Jun 2023,Monday
# Bihar CM Nitish Kumar on Monday said that the Aguwani Ghat-Sultanganj bridge in Bhagalpur that collapsed on Sunday was not constructed properly and that's why it is collapsing again and again. "I have instructed officials to take strict action against the construction company," the CM added. A section of the bridge collapsed on April 30, 2022, during a thunderstorm."""

# politics
# new_headline="Mukhtar Ansari gets life sentence in Congress leader's murder case"
# new_article="An MP/MLA court in Uttar Pradesh's Varanasi has sentenced jailed gangster-turned-politician Mukhtar Ansari to life imprisonment after he was held guilty in the 1991 Congress leader Awadhesh Rai's murder case. The court has also imposed a fine of ₹1 lakh on Ansari. Rai was shot dead outside his brother and former MLA Ajay Rai's residence in Varanasi."


# new_headline="Honda Elevate SUV to debut in India on June 6; what to expect, features, engine, price and rivals"
# new_article="""The Fronx was showcased at the Auto Expo 2023 along with the Jimny on January 12
# with bookings for both SUVs opening on the same day.The Honda Elevate will rival the Hyundai
# Creta, Kia Seltos, Maruti Suzuki Grand Vitara, Skoda Kushaq, Volkswagen Taigun and MG Astor.
# electric"""

# new_headline="2023 Hero HF Deluxe launched at Rs 60,760"
# new_article="""The 2023 Hero HF Deluxe now comes with tubeless tyres as standard on the self-start and i3s variants.Hero has updated its entry-level 100cc commuter, the HF Deluxe, and the bike now costs between Rs 60,760-67,208.
#
# 2023 Hero HF Deluxe engine, features, cycle parts
# For starters, the 2023 HF Deluxe now comes with tubeless tyres as standard on the self start and i3s (start/stop technology) equipped variants of the bike. A USB charger is also offered as an optional extra and the bike comes with a five year warranty and five free services as standard.

# Powering the Hero HF Deluxe is the air-cooled, 97cc, single-cylinder ‘Sloper’ mill that’s rated for 8hp and 8.05Nm of torque. This long-serving engine is now OBD-2 compliant plus E20 ready and is mated to a 4-speed gearbox. The engine is nestled inside a basic double cradle frame, suspended by a simple telescopic fork and 2-step adjustable twin shock absorber setup. With its 9.6-litre tank fully brimmed, the HF Deluxe weighs 112kg."
# """

# science
# new_headline="Satellite data reveal nearly 20,000 previously unknown deep-sea mountains"
# new_article="""Ship-mounted sonar reveals how Kelvin Seamount, off the coast of Massachusetts, rises from the seafloor (purple and blue denote low elevation while red is high). A new mapping technique based on satellite data has found thousands of previously unknown undersea mountains.
# WOODS HOLE OCEANOGRAPHIC INSTITUTION
# The number of known mountains in Earth’s oceans has roughly doubled.
# """

# new_headline="5,000 deep-sea animals new to science turned up in ocean records"
# new_article="""More than 5,000 animal species previously unknown to science live in a pristine part of the deep sea.
#
# Their home — called the Clarion-Clipperton Zone — sits in the central and eastern Pacific Ocean between Hawaii and Mexico. The zone is roughly twice the size of India, sits 4,000 to 6,000 meters deep and is largely a mystery, like much of the deep sea.
#
# In a new study, scientists amassed and analyzed more than 100,000 published records of animals found in the zone, with some records dating back to the 1870s. About 90 percent of species from these records were previously undescribed: There were only about 440 named species compared with roughly 5,100 without scientific names. Worms and arthropods make up the bulk of the undescribed creatures, but other animals found there include sponges, sea cucumbers and corals, the researchers report May 25 in Current Biology."""

# sports
# new_headline="More than half of the world’s largest lakes are drying up"
# new_article="""More than half of the world’s largest lakes shrank over the last three decades, researchers report in the May 19 Science.
#
# That’s a big problem for the people who depend on those lakes for drinking water and irrigation. Drying lakes also threaten the survival of local ecosystems and migrating birds, and can even give rise to insalubrious dust storms (SN: 4/17/23).
#
# “About one-quarter of the Earth’s population lives in these basins with lake water losses,” says surface hydrologist Fangfang Yao of the University of Virginia in Charlottesville."""

# new_headline="Ancient giant eruptions may have seeded nitrogen needed for life"
# new_article="""Millions of years ago, giant volcanic eruptions in what’s now Turkey and Peru each deposited millions of metric tons of nitrate on the surrounding land. That nutrient may have come from volcanic lightning, researchers reported April 24 at a meeting of the European Geosciences Union in Vienna.
#
# The discovery adds evidence to the idea that, early in Earth’s history, volcanoes could have provided some of the materials that made it possible for life to emerge, says volcanologist Erwan Martin of Sorbonne University in Paris."""

# sports
# new_headline="Pics: Virat Kohli, Rohit Sharma, Shubman Gill Pose In India's New Test Jersey"
# new_article="""With just a few days to go for the ICC World Test Championship final to begin, the Board of Control for Cricket In India (BCCI) announced a partnership with Adidas for India's kits across three formats. Ahead of the summit clash against Australia, the stalwarts of the Indian team posed in the new Test jersey, giving fans a glimpse of the new kit that the team will be wearing at the Oval, starting Wednesday. The likes of Virat Kohli, Rohit Sharma, Shubman Gill, Ravindra Jadeja and others could be seen posing in the new Test kit."""

# new_headline="With just a few days to go for the ICC World Test Championship final to begin, the Board of Control for Cricket In India (BCCI) announced a partnership with Adidas for India's kits across three formats. Ahead of the summit clash against Australia, the stalwarts of the Indian team posed in the new Test jersey, giving fans a glimpse of the new kit that the team will be wearing at the Oval, starting Wednesday. The likes of Virat Kohli, Rohit Sharma, Shubman Gill, Ravindra Jadeja and others could be seen posing in the new Test kit."
# new_article="""Ravindra Jadeja Or Ravichandran Ashwin At The Oval? Gavaskar Picks India's WTC Final Playing XISunil Gavaskar picked the playing XI for India in the World Test Championship (WTC) Final against Australia and it came with a couple of surprises.NDTV Sports DeskUpdated: June 05, 2023 08:05 AM ISTRead Time:2 min
# Ravindra Jadeja Or Ravichandran Ashwin At The Oval? Gavaskar Picks India's WTC Final Playing XI
# File photo of Sunil Gavaskar© Twitter
# Former Indian cricket team skipper Sunil Gavaskar picked his preferred playing XI for India in the World Test Championship (WTC) Final against Australia and it came with a couple of surprises. In the race for wicket-keeper, he picked KS Bharat to retain his position as he has been playing for the side regularly. Gavaskar also went with two spinners at the Oval and three pace bowlers but added that this strategy will work on a “bright day” and any change in playing conditions will change the combination.
#
# "I will talk about the batting and that will be Rohit Sharma and Shubman Gill as one-two. No. 3 is (Cheteshwar) Pujara, No. 4 is (Virat) Kohli, No. 5 is Ajinkya Rahane," Gavaskar said."""

data_test = pd.DataFrame({'news_headline': [new_headline] ,'news_article':[new_article]})
# special symbols removal
data_test['news_headline'] = data_test['news_headline'].apply(lambda headline: text_cleaning(headline))
data_test['news_article'] = data_test['news_article'].apply(lambda article: text_cleaning(article))
data_test = preprocessor.transform(data_test)
# Decode the encoded labels
news_genre = encoder_category.inverse_transform(svc.predict(data_test))
print("Prediction of a new datapoint : ",news_genre[0])
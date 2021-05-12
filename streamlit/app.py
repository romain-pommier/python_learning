import streamlit as st 
import pandas as pd 
import seaborn as sns 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from stop_words import get_stop_words
from joblib import dump, load
import re
import sys



data = pd.read_csv('./Data/labeled_data.csv')
data['tweet'] = data['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))

st.dataframe(data.head())
data['tweet'] = data['tweet'].replace('[\!,]', '', regex=True)
tweet = data['tweet']
st.dataframe(tweet.head())
# clf = make_pipeline(
# TfidfVectorizer(stop_words=get_stop_words('en')),
# OneVsRestClassifier(SVC(kernel='linear', probability=True))
# )

# clf = clf.fit(X=data['tweet'], y=data['class'])
# text = "I hate you, please die!"
# clf.predict_proba([text])[0]


# dump(clf, 'result.joblib')




clf = load('./result.joblib')
user_input = st.text_input("Your fucking text here !!")

# clf.predict_proba([user_input])[0]
st.write(pd.DataFrame(clf.predict_proba([user_input]), columns=['Hate', 'Offensive', 'Neutral']))

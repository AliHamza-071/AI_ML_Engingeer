#  Import Libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

#  Load and Merge Dataset
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("Real.csv")

fake_df['label'] = 0  # Fake
real_df['label'] = 1  # Real

df = pd.concat([fake_df, real_df])
df = df[['text', 'label']]  # Only use 'text' and 'label'
df = df.dropna()
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

#  Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Split Data Once
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Prepare Features for Naive Bayes and Random Forest
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#  Prepare Features for LSTM
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 300
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

#  Train and Evaluate Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_count, y_train)
nb_pred = nb.predict(X_test_count)
print("Naive Bayes:\n", classification_report(y_test, nb_pred))

#  Train and Evaluate Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_pred = rf.predict(X_test_tfidf)
print("Random Forest:\n", classification_report(y_test, rf_pred))

#  Train and Evaluate LSTM
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=3, batch_size=64, validation_data=(X_test_pad, y_test))

lstm_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
print("LSTM:\n", classification_report(y_test, lstm_pred))

#  Test Samples
sample_articles = [
    "The government has introduced a new policy for economic growth.",
    "Scientists claim the earth is flat in a new report.",
    "Breaking: Major celebrity scandal causes social media meltdown."
]

# Preprocess for all 3 models
clean_samples = [clean_text(t) for t in sample_articles]

# NB & RF
count_samples = vectorizer.transform(clean_samples)
tfidf_samples = tfidf_vectorizer.transform(clean_samples)

print("Naive Bayes Predictions:", nb.predict(count_samples))
print("Random Forest Predictions:", rf.predict(tfidf_samples))

# LSTM
seq_samples = tokenizer.texts_to_sequences(clean_samples)
pad_samples = pad_sequences(seq_samples, maxlen=max_len)
print("LSTM Predictions:", (model.predict(pad_samples) > 0.5).astype("int32").flatten())
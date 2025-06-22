# Import all necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download punkt_tab


# : Load the dataset
df = pd.read_csv('IMDB Dataset.csv')
#  Explore the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nData info:")
print(df.info())

print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

#  Text preprocessing
print("Applying text preprocessing...")

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Preprocess function without using a separate function
processed_reviews = []
for review in df['review']:
    # Convert to lowercase
    review = review.lower()

    # Remove HTML tags
    review = re.sub(r'<.*?>', '', review)

    # Remove special characters and numbers
    review = re.sub(r'[^a-zA-Z\s]', '', review)

    # Tokenize
    tokens = word_tokenize(review)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into string
    processed_review = ' '.join(filtered_tokens)
    processed_reviews.append(processed_review)

# Add processed reviews to dataframe
df['processed_review'] = processed_reviews
#  Inspect preprocessed data
print("\nSample processed review:")
print(df['processed_review'][0][:200], "...")
#  Prepare data for modeling
# Convert sentiment to binary (0: negative, 1: positive)
df['sentiment_binary'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the dataset into training and testing sets (80% train, 20% test)
X = df['processed_review']
y = df['sentiment_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature extraction using TF-IDF
print("Extracting features using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF features shape: {X_train_tfidf.shape}")

# Model training - Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test_tfidf)

# Calculate metrics
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

print(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, F1-Score: {lr_f1:.4f}")
print(classification_report(y_test, lr_predictions))

#  Model training - Naive Bayes
print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Make predictions
nb_predictions = nb_model.predict(X_test_tfidf)

# Calculate metrics
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)

print(f"Naive Bayes - Accuracy: {nb_accuracy:.4f}, F1-Score: {nb_f1:.4f}")
print(classification_report(y_test, nb_predictions))
# Model training - SVM
print("Training SVM...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# Make predictions
svm_predictions = svm_model.predict(X_test_tfidf)

# Calculate metrics
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)

print(f"SVM - Accuracy: {svm_accuracy:.4f}, F1-Score: {svm_f1:.4f}")
print(classification_report(y_test, svm_predictions))

#  Compare model performances
# Store results in a dictionary
results = {
    'Logistic Regression': {'accuracy': lr_accuracy, 'f1_score': lr_f1},
    'Naive Bayes': {'accuracy': nb_accuracy, 'f1_score': nb_f1},
    'SVM': {'accuracy': svm_accuracy, 'f1_score': svm_f1}
}
# Visualize the results
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]
f1_scores = [results[model]['f1_score'] for model in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy')
plt.bar(x + width/2, f1_scores, width, label='F1-Score')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

#  Test with sample reviews
# Get best model based on F1-score
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
print(f"Best performing model: {best_model_name} with F1-Score: {results[best_model_name]['f1_score']:.4f}")

# Depending on which model performed best
if best_model_name == 'Logistic Regression':
    best_model = lr_model
elif best_model_name == 'Naive Bayes':
    best_model = nb_model
else:
    best_model = svm_model

# Sample reviews for testing
sample_reviews = [
    "This movie was absolutely amazing! The acting was superb and the plot was engaging.",
    "Terrible film. Poor acting, bad script, and a waste of time. I wouldn't recommend it to anyone."
]

# Preprocess sample reviews
processed_samples = []
for review in sample_reviews:
    review = review.lower()
    review = re.sub(r'<.*?>', '', review)
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    tokens = word_tokenize(review)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    processed_review = ' '.join(filtered_tokens)
    processed_samples.append(processed_review)

# Transform sample reviews
sample_tfidf = tfidf_vectorizer.transform(processed_samples)

# Predict sentiment
predictions = best_model.predict(sample_tfidf)

print("\nSample review predictions using the best model:")
for i, review in enumerate(sample_reviews):
    sentiment = "Positive" if predictions[i] == 1 else "Negative"
    print(f"Review: {review[:50]}...")
    print(f"Predicted sentiment: {sentiment}\n")
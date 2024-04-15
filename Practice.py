# Import necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Make sure to also import nltk and download necessary resources if you're doing more advanced preprocessing
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(s):
    s = s.lower()  # Convert to lowercase
    s = re.sub(r'[^a-zA-Z\s]', '', s)  # Remove punctuation and numbers
    return s

# Load your dataset
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

# Apply the preprocessing to your text data
train_data['text'] = train_data['text'].apply(preprocess_text)
test_data['text'] = test_data['text'].apply(preprocess_text)

# Vectorization with TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['text'])

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, train_data['target'])

# Make predictions on the test set
predictions = model.predict(X_test_tfidf)

# Prepare your submission
submission = pd.DataFrame({'id': test_data['id'], 'target': predictions})
submission.to_csv('disaster_tweets_submission.csv', index=False)

print("Submission file is ready, Master Chief.")


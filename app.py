import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example data
texts = [
    "Win cash prize now",
    "Free offer call now",
    "Hi how are you",
    "Let's meet tomorrow"
]
labels = [1, 1, 0, 0]   # 1 = spam, 0 = ham

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

pipeline.fit(texts, labels)

pickle.dump(pipeline, open("trained_spam_classifier_model.pkl", "wb"))

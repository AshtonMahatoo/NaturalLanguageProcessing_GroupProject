import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

testSet = pd.read_csv('test_amazon.csv')
trainSet = pd.read_csv('train_amazon.csv')

vectorizer = vectorizer()
X_train = vectorizer.fit_transform(trainSet['text'])
X_test = vectorizer.transform(testSet['text'])

model = LogisticRegression()

model.fit(X_train, trainSet['label'])

y_pred = model.predict(X_test)

accuracy = accuracy_score(testSet['label'], y_pred)
print("Accuracy:", accuracy)
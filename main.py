import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import time

def basic(testSet, trainSet):
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(trainSet['text'])
    X_test = vectorizer.transform(testSet['text'])

    model = LogisticRegression()

    model.fit(X_train, trainSet['label'])

    y_pred = model.predict(X_test)

    return confusion_matrix(testSet['label'], y_pred)

def countVectorizer(testSet, trainSet):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(trainSet['text'])
    X_test = vectorizer.transform(testSet['text'])

    model = LogisticRegression()

    model.fit(X_train, trainSet['label'])

    y_pred = model.predict(X_test)

    return confusion_matrix(testSet['label'], y_pred)

testSet = pd.read_csv('test_amazon.csv')
trainSet = pd.read_csv('train_amazon.csv')
start_time = time.time()

CM = basic(testSet, trainSet)

end_time = time.time()
elapsed_time = end_time - start_time

TP = CM[1][1]
TN = CM[0][0]
FP = CM[0][1]
FN = CM[1][0]

accuracy = ((TP + TN)/(TP + TN + FP + FN))
print("True Positives: ", TP)
print("True Negative: ", TN)
print("False Positives: ", FP)
print("False Negatives: ", FN)
print("OVerall Accuracy: ", accuracy)

print(f"Elapsed time: {elapsed_time} seconds")
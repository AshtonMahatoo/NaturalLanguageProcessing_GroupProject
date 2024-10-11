import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import time

def logistic_regression_model(vectorizer, trainSet, testSet, solver='lbfgs', max_iter=200):

    X_train = vectorizer.fit_transform(trainSet['text'])
    X_test = vectorizer.transform(testSet['text'])

    model = LogisticRegression(solver=solver, max_iter=max_iter)

    start_time = time.time()
    model.fit(X_train, trainSet['label'])
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    cm = confusion_matrix(testSet['label'], y_pred)
    accuracy = accuracy_score(testSet['label'], y_pred)

    return cm, accuracy, training_time, inference_time

def evaluate_model(cm):
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return TP, TN, FP, FN, accuracy

# Experiment with training sizes - 40k and 80k
def experiment_with_size(trainSet, testSet, vectorizer_type='tfidf', solver='lbfgs', max_iter=200, sizes=[40000, 80000]):
    vectorizer = TfidfVectorizer() if vectorizer_type == 'tfidf' else CountVectorizer()
    
    for size in sizes:
        subset_trainSet = trainSet[:size]
        print(f"\nTraining with {size} reviews - ")
        
        cm, accuracy, training_time, inference_time = logistic_regression_model(vectorizer, subset_trainSet, testSet, solver, max_iter)

        TP, TN, FP, FN, _ = evaluate_model(cm)
        print("True Positives: ",TP)
        print("True Negatives: ",TN)
        print("False Positives: ",FP)
        print("False Negatives: ",FN)
        print("Overall Accuracy: ",accuracy)
        print(f"Training Time: {training_time} seconds")
        print(f"Inference Time: {inference_time} seconds")

# Experiment with the full dataset
def experiment_full_dataset(trainSet, testSet, vectorizer_type='tfidf', solver='lbfgs', max_iter=200):
    vectorizer = TfidfVectorizer() if vectorizer_type == 'tfidf' else CountVectorizer()
    
    print(f"\nTraining with the full dataset ({len(trainSet)} reviews) - ")

    # Train and evaluate the logistic regression on the full dataset
    cm, accuracy, training_time, inference_time = logistic_regression_model(vectorizer, trainSet, testSet, solver, max_iter)

    TP, TN, FP, FN, _ = evaluate_model(cm)
    print("True Positives: ",TP)
    print("True Negatives: ",TN)
    print("False Positives: ",FP)
    print("False Negatives: ",FN)
    print(f"Overall Accuracy: {accuracy}")
    print(f"Training Time: {training_time} seconds")
    print(f"Inference Time: {inference_time} seconds")

testSet = pd.read_csv('test_amazon.csv')
trainSet = pd.read_csv('train_amazon.csv')

# Experiments using TF-IDF and different solvers on different dataset sizes
print("\nExperiment 1 - TF-IDF with lbfgs solver")
experiment_with_size(trainSet, testSet, vectorizer_type='tfidf', solver='lbfgs', max_iter=200, sizes=[40000, 80000])

print("\nExperiment 2 - TF-IDF with liblinear solver")
experiment_with_size(trainSet, testSet, vectorizer_type='tfidf', solver='liblinear', max_iter=200, sizes=[40000, 80000])

# Experiments using CountVectorizer and different solvers on different dataset sizes
print("\nExperiment 3 - CountVectorizer with lbfgs solver")
experiment_with_size(trainSet, testSet, vectorizer_type='count', solver='lbfgs', max_iter=200, sizes=[40000, 80000])

print("\nExperiment 4 - CountVectorizer with liblinear solver")
experiment_with_size(trainSet, testSet, vectorizer_type='count', solver='liblinear', max_iter=200, sizes=[40000, 80000])

# Full dataset experiments with TF-IDF
print("\nExperiment 5 - TF-IDF with lbfgs solver on full dataset")
experiment_full_dataset(trainSet, testSet, vectorizer_type='tfidf', solver='lbfgs', max_iter=200)

print("\nExperiment 6 - TF-IDF with liblinear solver on full dataset")
experiment_full_dataset(trainSet, testSet, vectorizer_type='tfidf', solver='liblinear', max_iter=200)

# Full dataset experiments with CountVeectorizer
print("\nExperiment 7 - CountVectorizer with lbfgs solver on full dataset")
experiment_full_dataset(trainSet, testSet, vectorizer_type='count', solver='lbfgs', max_iter=200)

print("\nExperiment 8 - CountVectorizer with liblinear solver on full dataset")
experiment_full_dataset(trainSet, testSet, vectorizer_type='count', solver='liblinear', max_iter=200)
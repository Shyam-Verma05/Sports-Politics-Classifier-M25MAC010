from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_svm(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = SVC()
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    print("SVM Accuracy:", accuracy_score(y_test, preds))

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import pickle

# Prepare data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

sgd_clf = SGDClassifier(verbose=1)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
print(classification_report(y_test, y_pred))
model_file = "mutil_sgd_clf.pkl"
pickle.dump(sgd_clf, open(model_file, 'wb'))
print('Save model done :{}'.format(model_file))

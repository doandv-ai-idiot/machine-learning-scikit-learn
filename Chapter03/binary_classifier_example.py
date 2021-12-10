from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
import joblib
import pickle

# Prepare data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')  # True for all 5s, False for all other digits.
y_test_5 = (y_test == '5')
# Init model
sgd_clf = SGDClassifier(verbose=0)
sgd_clf.fit(X_train, y_train_5)
# Eval
y_pred = sgd_clf.predict(X_test)
# Report
print(classification_report(y_test_5, y_pred))
# Measure accuracy
skfolds = StratifiedKFold(n_splits=3)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train.values[train_index]
    y_train_folds = y_train_5.values[train_index]
    X_test_folds = X_train.values[test_index]
    y_test_folds = y_train_5.values[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_corrects = sum(y_pred == y_test_folds)
    print(n_corrects / len(y_pred))
# Using cross val  score in sklearn
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=5, scoring='accuracy'))
# using confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=5)
print(confusion_matrix(y_train_5, y_train_pred))
# precision and recall f1 score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))

# Save model
model_name = "sgd_clf.sav"
joblib.dump(sgd_clf, open(model_name, 'wb'))
pickle.dump(sgd_clf, open(model_name, 'wb'))
print('Save done with model path:{}'.format(model_name))

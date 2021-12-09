from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import OneHotEncoder

mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
x, y = mnist['data'].values, mnist['target'].values
x_train, y_train, x_test, y_test = x[:60000], y[:60000], x[60000:], y[60000:]


def show_data_image(x, i):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    some_digit = x.values[i]
    some_digit_img = some_digit.reshape(28, 28)
    plt.imshow(some_digit_img, cmap=mpl.cm.get_cmap('binary'), interpolation='nearest')
    plt.axis('off')
    plt.show()


# Binary Classifier
def binary_classifier():
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import classification_report
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    # y_train_5 = y_train_5.astype(np.uint8)
    # y_test_5 = y_test_5.astype(np.uint8)
    sgd_classifier = SGDClassifier(random_state=42)
    sgd_classifier.fit(x_train, y_train_5)
    predicted = sgd_classifier.predict(x_test)
    print(classification_report(y_test_5, predicted))


if __name__ == '__main__':
    binary_classifier()

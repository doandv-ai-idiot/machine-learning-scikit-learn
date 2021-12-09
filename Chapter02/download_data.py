import os
import tarfile
import pandas as pd
from six.moves import urllib
import numpy as np
from sklearn.model_selection import train_test_split

# CONFIG

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test_custom(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__':
    # fetch_data()
    housing = load_housing_data()
    # housing.head()
    # housing.info()
    # print(housing.describe())
    # Plot histogram housing data
    import matplotlib.pyplot as plt

    housing.hist(bins=50, figsize=(20, 15))
    plt.show()
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                 s=housing['population'] / 100, label='population', figsize=(10, 7),
                 c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.legend()
    plt.show()
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    print(len(train_set))
    print(len(test_set))

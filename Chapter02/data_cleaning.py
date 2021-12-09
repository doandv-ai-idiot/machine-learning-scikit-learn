import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

HOUSING_PATH = os.path.join("datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def fill_na_null():
    housing = load_housing_data()
    print(len(housing))
    # Clean miss data of feature total bedroom
    # Option 1
    housing.dropna(subset=['total_bedrooms'])
    # Option 2
    housing.drop('total_bedrooms', axis=1)
    # Option 3 : fill null by median
    median = housing['total_bedrooms'].median()
    housing.fillna(median, inplace=True)
    housing.info()
    print(len(housing))


def clear_null_na_by_impute():
    housing = load_housing_data()
    print(len(housing))
    imputer = SimpleImputer(strategy='median')
    # Drop column have value is string
    housing_num = housing.drop('ocean_proximity', axis=1)
    imputer.fit(housing_num)
    print(imputer.statistics_)
    print(housing_num.median().values)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    print(len(housing_tr))


def clean_data_with_column_value_text():
    housing = load_housing_data()
    print(len(housing))
    housing_cat = housing[['ocean_proximity']]
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoder = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoder[:10])
    print(ordinal_encoder.categories_)
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot.toarray())
    print(cat_encoder.categories_)




if __name__ == '__main__':
    clean_data_with_column_value_text()

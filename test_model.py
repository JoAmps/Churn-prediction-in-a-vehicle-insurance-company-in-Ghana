import pandas as pd
import pytest
from clean_data import load_data,cleaned_data
from joblib import load
from data_preprocess import process_data
from model import get_cat_features, split_data

@pytest.fixture
def data():
    """
    Obtain data
    """
    df = load_data("veh_invest.csv")
    df = cleaned_data(df)
    return df

def test_null(data):
    """
    Check data has no null values
    """
    assert data.shape == data.dropna().shape

def test_split_data(data):
    """
    check train data has more data than test data
    """
    train, test = split_data(data)
    assert len(train) > len(test)    

def test_process_train(data):
    """
    Check train data has same number of rows for X and y
    """
    X_train, y_train, _, _ ,_= process_data(data,training=True,label='churn',
    cat_features=get_cat_features())
    assert X_train.shape[0] == y_train.shape[0]  



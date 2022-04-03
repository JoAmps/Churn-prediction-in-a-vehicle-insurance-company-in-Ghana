import pandas as pd
import pytest
from clean_data import load_data,cleaned_data
from joblib import load
from data_preprocess import process_data
from model import get_cat_features

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
    Data is assumed to have no null values
    """
    assert data.shape == data.dropna().shape

def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """
    ohe = load("ohe.joblib")
    lb = load("lb.joblib")
    scaler=load('scaler.joblib')

    X_test, y_test, _, _ = process_data(data,training=False,label='churn',
    cat_features=get_cat_features(),ohe=ohe, lb=lb,scaler=scaler)

    assert len(X_test) == len(y_test)  
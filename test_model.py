import pytest
from clean_data import load_data
from functions.data_preprocess import process_data
from functions.model import get_cat_features, split_data, model_predictions
from joblib import load


@pytest.fixture
def data():
    """
    Obtain data
    """
    df = load_data("datasets/raw_data.csv")
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
    X_train, y_train, _, _, _ = process_data(
        data, training=True, label='churn', cat_features=get_cat_features())
    assert X_train.shape[0] == y_train.shape[0]


def test_process_test(data):
    """
    Check test data has same number of rows for X and y
    """
    ohe = load("outputs/ohe.joblib")
    lb = load("outputs/lb.joblib")
    scaler = load("outputs/scaler.joblib")
    X_test, y_test, _, _, _ = process_data(
        data, training=False, label='churn', cat_features=get_cat_features(),
        lb=lb, ohe=ohe, scaler=scaler)
    assert X_test.shape[0] == y_test.shape[0]


def test_predictions_data(data):
    """
    Check if test data has the same length as predictions
    """
    ohe = load("outputs/ohe.joblib")
    lb = load("outputs/lb.joblib")
    scaler = load("outputs/scaler.joblib")
    model_object = load("outputs/model.joblib")
    X_test, _, _, _, _ = process_data(
        data, training=False, label='churn', cat_features=get_cat_features(),
        lb=lb, ohe=ohe, scaler=scaler)
    predictions = model_predictions(X_test, model_object)
    assert len(X_test) == len(predictions)

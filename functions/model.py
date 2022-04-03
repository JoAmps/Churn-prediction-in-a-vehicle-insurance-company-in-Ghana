from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def split_data(data):
    """
    Splits the data into training and testing
    Inputs
    -------
    data : pandas dataframe
           The cleaned data
    Returns
    -------
    train
        train data for training
    test
        test data for validation
    """
    try:
        train, test = train_test_split(data, test_size=0.2, random_state=0)
        logging.info('SUCCESS!:Data split successfully')
        return train, test
    except BaseException:
        logging.info('Error!:Error whiles splitting data')


def get_cat_features():
    cat_features = [
        'city',
        'type_of_plan',
        'highest_level_education',
        'work_status',
        'sex',
        'relationship_status',
        'reachability',
        'type_of_vehicle']
    return cat_features


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns the
    trained model.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    try:
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        logging.info('SUCCESS!:Model trained and saved')
        return model
    except BaseException:
        logging.info('ERROR!:Model not trained and not saved')


def model_predictions(X_test, model):
    """
    Performs prediction on the independent testing data using the trained
    machine learning model
    Inputs
    ------
    X_test : np.array
             Testing data
    y_test : np.array
             Test labels
    Returns
    -------
    predictions : int
    """
    try:
        predictions = model.predict(X_test)
        logging.info('SUCCESS!:Model predictions generated')
        return predictions
    except BaseException:
        logging.info('ERROR!:Model predictions not generated')


def compute_metrics(y, predictions):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    try:
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        logging.info('SUCCESS: Model scoring completed')
        return precision, recall, f1
    except BaseException:
        logging.info('ERROR: Error occurred when scoring Models')


def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

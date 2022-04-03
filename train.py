
import logging
from functions.data_preprocess import process_data
from clean_data import load_data
from functions.model import train_model, \
    compute_metrics, model_predictions, split_data, get_cat_features
from joblib import dump

logging.basicConfig(
    filename='./outputs/process.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    df = load_data('datasets/cleaned_data.csv')
    train, test = split_data(df)
    X_train, y_train, lb, ohe, scaler = process_data(
        train, training=True, label='churn', cat_features=get_cat_features())
    X_test, y_test, lb_t, ohe_t, scaler_t = process_data(
        test, training=False, label='churn', cat_features=get_cat_features(), ohe=ohe, lb=lb, scaler=scaler)
    dump(lb_t, './outputs/lb.joblib')
    dump(ohe_t, './outputs/ohe.joblib')
    dump(scaler_t, './outputs/scaler.joblib')
    model = train_model(X_train, y_train)
    predictions = model_predictions(X_test, model)
    precision, recall, f1 = compute_metrics(y_test, predictions)
    model_scores = []
    scores = "precision: %s " \
        "recall: %s f1: %s" % (precision, recall, f1)
    model_scores.append(scores)
    with open('./outputs/model_metrics.txt', 'w') as out:
        for score in model_scores:
            out.write(score)

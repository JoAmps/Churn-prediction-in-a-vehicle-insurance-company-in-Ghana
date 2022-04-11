
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
    df = load_data()
    train, test = split_data(df)
    X_train, y_train, lb, ohe, scaler = process_data(
        train, training=True, label='churn', cat_features=get_cat_features())
    X_test, y_test, lb, ohe, scaler = process_data(
        test, training=False, label='churn', cat_features=get_cat_features(),
        lb=lb, ohe=ohe, scaler=scaler)
    dump(lb, './outputs/lb.joblib')
    dump(ohe, './outputs/ohe.joblib')
    dump(scaler, './outputs/scaler.joblib')
    model = train_model(X_train, y_train)
    dump(model, './outputs/model.joblib')
    predictions = model_predictions(X_test, model)
    precision, recall, f1 = compute_metrics(y_test, predictions)
    with open("./outputs/recall_score.txt", "w") as file:
        file.write(recall.astype('str'))
    model_scores = []
    scores = "precision: %s " \
        "recall: %s f1: %s" % (precision, recall, f1)
    model_scores.append(scores)
    with open('./outputs/model_metrics.txt', 'w') as out:
        for score in model_scores:
            out.write(score)

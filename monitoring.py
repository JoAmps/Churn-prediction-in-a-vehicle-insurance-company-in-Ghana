import os
import glob
import numpy as np
import ast
import logging
from joblib import load
from functions import data_preprocess
from functions.model import compute_metrics, model_predictions,\
    get_cat_features
import pandas as pd


logging.basicConfig(
    filename='./outputs/monitoring.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def monitor_model_and_data():
    input_data = glob.glob('datasets/*.csv')
    input_data = glob.glob('datasets/*.csv')
    current_data = glob.glob('datasets/cleaned_data*.csv')
    latest_data = [max(input_data, key=os.path.getctime)]
    if latest_data == current_data:
        logging.info('no new data available')
        logging.info('end process')
        exit
    else:
        logging.info('New_data_available')
        logging.info(
            'Therefore running training\
        script on new data')
        model_object = load("outputs/model.joblib")
        ohe = load("outputs/ohe.joblib")
        lb = load("outputs/lb.joblib")
        scaler = load("outputs/scaler.joblib")

        df_new = pd.concat([pd.read_csv(i, index_col=[0])
                           for i in (input_data)])

        X, y, _, _, _ = data_preprocess.process_data(
            df_new,
            training=False, label='churn',
            cat_features=get_cat_features(),
            lb=lb, ohe=ohe, scaler=scaler)

        predictions = model_predictions(X, model_object)
        _, new_recall_score, _ = compute_metrics(y, predictions)
        with open('outputs/recall_score.txt', 'r') as fp:
            current_recall_score = ast.literal_eval(fp.read())

        with open("./outputs/new_recall_score.txt", "w") as file:
            file.write(new_recall_score.astype('str'))

        if new_recall_score >= np.max(current_recall_score) is True:
            logging.info(
                'New recall score greater than or equal to previous score,\
                model_drift not occurred')
            logging.info('end process')
            exit
        else:
            logging.info(
                'Current recall score less than previous score,\
                     model_drift may have occurred')
            logging.info('further_investigation_warranted')
            exit


if __name__ == '__main__':
    monitor_model_and_data()

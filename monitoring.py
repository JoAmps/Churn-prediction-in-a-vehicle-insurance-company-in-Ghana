import os
import glob
import numpy as np
import ast
import logging

logging.basicConfig(
    filename='./outputs/monitoring.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def monitor_model_and_data():
    input_data = glob.glob('datasets/*.csv')
    current_data = glob.glob('datasets/cleaned_data*.csv')
    latest_data = [max(input_data, key=os.path.getctime)]
    if latest_data == current_data:
        logging.info('no new data available')
        logging.info('end process')
        exit
    else:
        logging.info('new_data_available')
        with open('outputs/recall_score.txt', 'r') as fp:
            previous_recall_score = ast.literal_eval(fp.read())
        logging.info(
            'New data available, therefore running training\
                 script on new data')
        os.system('python clean_data.py')
        os.system('python train.py')
        with open('outputs/recall_score.txt', 'r') as fp:
            current_recall_score = ast.literal_eval(fp.read())
        if current_recall_score < np.max(previous_recall_score) is not True:
            logging.info(
                'Current recall score not less than previous score,\
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

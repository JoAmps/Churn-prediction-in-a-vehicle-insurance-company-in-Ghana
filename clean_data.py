import pandas as pd
import logging


logging.basicConfig(
    filename='./outputs/process.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def load_data(path):
    try:
        df = pd.read_csv(path, index_col=[0])
        return df
        logging.info('SUCCESS: Data imported succesfully')
    except BaseException:
        logging.info('ERROR: Data not imported')


def save_data(df):
    try:
        logging.info('SUCCESS: Data saved succesfully')
        return df.to_csv('./datasets/cleaned_data.csv')
    except BaseException:
        logging.info('ERROR: Data not saved')     


if __name__ == '__main__':
    df = load_data("datasets/raw_data.csv")
    save_data(df)

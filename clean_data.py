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
        logging.info('SUCCESS: Data imported succesfully')
        return df
    except BaseException:
        logging.info('ERROR: Data not imported')


if __name__ == '__main__':
    df = load_data("datasets/raw_data.csv")

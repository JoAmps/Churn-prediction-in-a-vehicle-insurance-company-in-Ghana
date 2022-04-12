import pandas as pd
import logging
import glob
import os


logging.basicConfig(
    filename='./outputs/process.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


input_data = glob.glob('datasets/*.csv')
current_data = [max(glob.glob('datasets/*.csv'), key=os.path.getctime)]


def load_data():
    try:
        df = pd.concat([pd.read_csv(i, index_col=[0]) for i in (current_data)])
        df = df.drop(columns=df.filter(like='Unnamed'))
        logging.info('SUCCESS: Data imported succesfully')
        return df
    except BaseException:
        logging.info('ERROR: Data not imported')


def save_data(df):
    try:
        logging.info('SUCCESS: Data saved succesfully')
        return df.to_csv('./datasets/cleaned_data.csv')
    except BaseException:
        logging.info('ERROR: Data not saved')


if __name__ == '__main__':
    df = load_data()
    save_data(df)

import pandas as pd
import logging
import glob
import os

logging.basicConfig(
    filename='./outputs/process.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


input_data = glob.glob(f'datasets/*.csv')
current_data=glob.glob(f'datasets/cleaned_data*.csv')

def load_data():
    try:
        df=pd.concat([pd.read_csv(i) for i in (current_data)])
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

import pandas as pd
import logging


logging.basicConfig(
    filename='log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info('SUCCESS: Data imported succesfully')
        return df
    except BaseException:
        logging.info('ERROR: Data not imported')

def cleaned_data(df):
    try:
        df.drop(columns=['Unnamed: 0'])
        logging.info('SUCCESS: Unneccesary columns removed')
        df.to_csv('./datasets/cleaned_data.csv')
        return df   
    except BaseException:
        logging.info('ERROR: Columns removal error')    

if __name__ == '__main__':
    df = load_data("datasets/raw_data.csv")
    cleaned_data(df)
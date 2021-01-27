#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import urllib
import pandas as pd
from pathlib import Path

# Main source for the training data
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
# Path to where this script lives
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Local file
DATA_PATH = ROOT_DIR / 'data'
DATA_FILE = DATA_PATH / 'OxCGRT_latest.csv'
INPUT_DATA = DATA_PATH / 'input_data.csv'
# Model file
MODEL_WEIGHTS_FILE = ROOT_DIR / 'standard_predictor' / 'models' / 'trained_model_weights.h5'
# Surrogate Dataset
SURROGATE_DATA_FILE = DATA_PATH / 'surrogate_dataset.csv'

NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

ID_COLS = ['CountryName',
           'RegionName',
           'Date',
           'GeoID',
           'NewCases']

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}

def add_geo_id(df):
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
    return df

# Function that performs basic loading and preprocessing of historical df
def prepare_historical_df():

    # Download data if it we haven't done that yet.
    if not os.path.exists(DATA_FILE):
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    # Load raw historical data
    df = pd.read_csv(DATA_FILE,
                  parse_dates=['Date'],
                  encoding="ISO-8859-1",
                  error_bad_lines=False)
    df['RegionName'] = df['RegionName'].fillna("")

    # Add GeoID column for easier manipulation
    df = add_geo_id(df)

    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing IPs by assuming they are the same as previous day
    for ip_col in IP_MAX_VALUES:
        df.update(df.groupby('GeoID')[ip_col].ffill().fillna(0))

    return df
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

import pandas as pd
import numpy as np

import os
import urllib.request

from smt.sampling_methods import LHS
from validation.scenario_generator import generate_scenario
from standard_predictor.xprize_predictor import XPrizePredictor

# Model file
MODEL_WEIGHTS_FILE = "standard_predictor/models/trained_model_weights.h5"
# Main source for the training data
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
# Local file
DATA_FILE = 'data/OxCGRT_latest.csv'
DATA_PREDICTION = 'data/ips_predict.csv'

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
           'Date']

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

# Setting Parameters
nsample = 1000
days = 90
lag_size= 7
lag = int(days / lag_size) + 1
start_date_str = "2020-12-10"
end_date_str = "2020-1-26"
keys = NPI_COLUMNS + ID_COLS

# Generation of random NPI
xlimits = np.array(list(zip([0] * len(IP_MAX_VALUES), [x + 1 for x in list(IP_MAX_VALUES.values())])))
sampling = LHS(xlimits=xlimits, criterion='center')
npi = list(map(int, (list(sampling(1)[0]))))

# Read/Download historical data
if not os.path.exists('data'):
    os.mkdir('data')
urllib.request.urlretrieve(DATA_URL, DATA_FILE)
raw_df = pd.read_csv(DATA_FILE,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str},
                     error_bad_lines=False)

# Creating Dataset for Standard Predictor
df_historical = generate_scenario(start_date_str, end_date_str, raw_df, countries=None, scenario='Historical')
ips_dict = {
    ''
}
ips_df =
ips_df.to_csv(DATA_PREDICTION, sep=',')

# nsamples = 1000
# days = 90
# window = 7
# lag = int(days/7) + 1
# X = []
#
# def generate_ip_sample(nsamples, lag):
#     xlimits = np.array(list(zip([0] * len(IP_MAX_VALUES), [x + 1 for x in list(IP_MAX_VALUES.values())])))
#     sampling = LHS(xlimits=xlimits, criterion='center')
#     for _ in range(nsamples):
#         IP = np.array(sampling(lag), dtype='int32')
#         X.append(IP)

# df_historical = pd.read_csv(DATA_FILE)
# geos = pd.unique(df_historical['CountryCode'])
# for geo in geos:
#
# predictor = XPrizePredictor(MODEL_WEIGHTS_FILE, DATA_FILE)
# preds_df = predictor.predict("2021-01-02", "2021-01-10", DATA_FILE)
# print(preds_df[preds_df['CountryName'] == 'Italy'])



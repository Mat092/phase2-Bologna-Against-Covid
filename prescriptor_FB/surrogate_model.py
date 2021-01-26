#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import datetime
import utils

import pandas as pd
import numpy as np

import os
import urllib.request

from smt.sampling_methods import LHS
from validation.scenario_generator import generate_scenario
from standard_predictor.xprize_predictor import XPrizePredictor

# Setting Parameters
nsamples = 1000
days = 90
lag_size= 7
lag = int(days / lag_size) + 1
start_date_str = "2020-12-10"
end_date_str = "2021-1-1"
keys = utils.ID_COLS + utils.NPI_COLUMNS

# Generation of random NPI
def generate_random_sample(lag):
    xlimits = np.array(list(zip([0] * len(utils.IP_MAX_VALUES), [x + 1 for x in list(utils.IP_MAX_VALUES.values())])))
    sampling = LHS(xlimits=xlimits, criterion='center')
    npi = [list(map(int, x)) for x in sampling(lag_size)]
    return npi

# Read/Download historical data
raw_df = utils.prepare_historical_df()
# Creating Dataset for Standard Predictor
start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
df_historical = raw_df[(raw_df['Date'] >= start_date) &
                       (raw_df['Date'] <= end_date)]

df_historical = df_historical[keys]

geoids = pd.unique(df_historical['GeoID'])

predictor = XPrizePredictor(utils.MODEL_WEIGHTS_FILE, utils.DATA_FILE)
new_cases = predictor.predict(start_date_str, end_date_str, utils.DATA_FILE)

sys.exit()
data = []

first_day = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
second_day = datetime.datetime.strptime(end_date_str, '%Y-%m-%d') + \
             datetime.timedelta(days=1)

# For each country and reagion
for gid in geoids:
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    first_day_cases = raw_df[(raw_df['GeoID'] == gid) &
                             (raw_df['Date'] == first_day)]['NewCases'].values[0]
    npis = generate_random_sample(lag=lag)
    for npi in npis:
        for _ in range(lag_size):
            end_date += datetime.timedelta(days=1)
            # date = str(end_date).split(' ')[0].replace('-', '')
            data.append(
                {'CountryName': gid.split('__')[0],
                 'RegionName': gid.split('__')[1],
                 'Date': end_date,
                 'GeoID': gid,
                 'C1_School closing': npi[0],
                 'C2_Workplace closing': npi[1],
                 'C3_Cancel public events': npi[2],
                 'C4_Restrictions on gatherings': npi[3],
                 'C5_Close public transport': npi[4],
                 'C6_Stay at home requirements': npi[5],
                 'C7_Restrictions on internal movement': npi[6],
                 'C8_International travel controls': npi[7],
                 'H1_Public information campaigns': npi[8],
                 'H2_Testing policy': npi[9],
                 'H3_Contact tracing': npi[10],
                 'H6_Facial Coverings': npi[11],
                 'NewCases': np.nan # if end_date != second_day else first_day_cases
                 }
            )

df_preds = pd.DataFrame(data)
ips_df = pd.concat([df_historical, df_preds])
# print(ips_df[ips_df['CountryName'] == 'Italy'])
# sys.exit()
start_date_prediction_str = str(datetime.datetime.strptime(end_date_str, '%Y-%m-%d') +
                                datetime.timedelta(days=1)).split(' ')[0]
# start_date_prediction_str = end_date_str
end_date_prediction_str = str(datetime.datetime.strptime(end_date_str, '%Y-%m-%d') +
                              datetime.timedelta(days=lag_size*lag)).split(' ')[0]
predictor = XPrizePredictor(utils.MODEL_WEIGHTS_FILE, utils.DATA_FILE)
new_cases = predictor.predict_from_df(start_date_prediction_str, end_date_prediction_str, ips_df)
print(new_cases[new_cases['CountryName'] == 'Italy'])
# print(new_cases[new_cases['CountryName'] == 'Italy'])
# new_cases = predictor.predict("2021-01-02", "2021-01-10", utils.DATA_PREDICTION)
# ips_df['NewCases'] = new_cases
#
# npis = pd.unique(ips_df[ips_df.columns != 'NewCases'])
# X = npis.values.flatten()
# y = list(ips_df['NewCases'])[-1]



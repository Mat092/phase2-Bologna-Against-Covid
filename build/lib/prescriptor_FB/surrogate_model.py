#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

import utils

import pandas as pd
import numpy as np

import os
import urllib.request

from smt.sampling_methods import LHS
from validation.scenario_generator import generate_scenario
from standard_predictor.xprize_predictor import XPrizePredictor

# Setting Parameters
nsample = 1000
days = 90
lag_size= 7
lag = int(days / lag_size) + 1
start_date_str = "2020-12-10"
end_date_str = "2020-1-26"
keys = utils.ID_COLS + utils.NPI_COLUMNS

# Generation of random NPI
xlimits = np.array(list(zip([0] * len(utils.IP_MAX_VALUES), [x + 1 for x in list(utils.IP_MAX_VALUES.values())])))
sampling = LHS(xlimits=xlimits, criterion='center')
npi = list(map(int, (list(sampling(1)[0]))))

# Read/Download historical data
raw_df = utils.prepare_historical_df()

# Creating Dataset for Standard Predictor
df_historical = generate_scenario(start_date_str, end_date_str, raw_df, countries=None, scenario='Historical')
print(df_historical)
sys.eixit()
# countries = pd.unique(df_historical['CountryName'])
# ips_df =
# ips_df.to_csv(utils.DATA_PREDICTION, sep=',')

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



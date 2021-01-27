#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, csv
import datetime, time
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
days = 7
lag_size= 7
lag = int(days / lag_size)
start_date_str = "2021-1-1"
end_date_str = "2021-1-8"
keys = utils.ID_COLS + utils.NPI_COLUMNS

# Generation of random NPI
class IPGenerator(object):

    def __init__(self, lag, random_state=125):
        self.lag = lag
        self.random_state = random_state
        self.xlimits = np.array(list(zip([0] * len(utils.IP_MAX_VALUES), [x + 1 for x in list(utils.IP_MAX_VALUES.values())])))
        self.sampling = LHS(xlimits=self.xlimits, criterion='center', random_state=self.random_state)

    def sample(self):
        npi = [list(map(int, x)) for x in self.sampling(self.lag)]
        return npi

def generate_surrogate_dataset():
    t0 = round(time.time())
    # Read/Download historical data
    raw_df = utils.prepare_historical_df()
    geoids = raw_df.GeoID.unique()
    for _ in range(nsamples):
        data = []
        for gid in geoids:
            generator = IPGenerator(lag)
            npis = generator.sample()
            date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
            for npi in npis:
                for _ in range(lag_size):
                    dict_geo  = {
                                'CountryName': gid.split('__')[0],
                                'RegionName': gid.split('__')[1],
                                'Date': date,
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
                                }
                    date += datetime.timedelta(days=1)
                    data.append(dict_geo)
        df_ips = pd.DataFrame(data)
        df_ips.to_csv(utils.INPUT_DATA, sep=',')
        predictor = XPrizePredictor(utils.MODEL_WEIGHTS_FILE, utils.DATA_FILE)
        pred_df = predictor.predict(start_date_str, end_date_str, utils.INPUT_DATA)
        t1 = round(time.time())
        # print(pred_df)
        print((t1 - t0)/60)

        pred_df = pred_df.fillna('')
        pred_df = utils.add_geo_id(pred_df)
        geoids = pred_df.GeoID.unique()
        df_ips = df_ips.fillna('')
        df_ips = utils.add_geo_id(df_ips)
        for gid in geoids:
            df_entry = df_ips[df_ips['GeoID'] == gid][utils.NPI_COLUMNS].drop_duplicates()
            df_entry['GeoID'] = gid
            df_entry['NewCases'] = pred_df[pred_df['GeoID'] == gid]['PredictedDailyNewCases'].values[-1]
            df_entry.to_csv(utils.SURROGATE_DATA_FILE, mode='a', header=False, sep=',')


if __name__ == '__main__':
    generate_surrogate_dataset()
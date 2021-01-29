#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import datetime
import utils

import pandas as pd

from pyDOE import lhs
from standard_predictor.xprize_predictor import XPrizePredictor


# Generation of random NPI
class DatasetGenerator():

    def __init__(self, nsamples, lag_size, random_state=42):
        self.nsamples = nsamples
        self.lag_size = lag_size
        self.random_state = random_state
        self.raw_df = utils.prepare_historical_df()

    def _npi_LHS(self):
        samples = lhs(len(utils.NPI_COLUMNS), self.nsamples)
        var_range = [x + 1 for x in utils.IP_MAX_VALUES.values()]
        configs = []
        for sample in samples:
            configs.append([int(var_range[j] * var) for j, var in enumerate(sample)])
        return configs

    def generate_surrogate_dataset(self, geo_id, start_date_str):
        dataset = []
        # Read/Download historical data
        for npi in self._npi_LHS():
            # create predictionn input data
            data = utils.create_input_data(geo_id, npi, start_date_str, self.lag_size)
            end_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d') + datetime.timedelta(days=6)
            end_date_str = str(end_date).split(' ')[0]

            # save input data
            df_ips = pd.DataFrame(data)
            df_ips.to_csv(utils.INPUT_DATA, sep=',')

            # predict
            predictor = XPrizePredictor(utils.MODEL_WEIGHTS_FILE, utils.DATA_FILE)
            df_pred = predictor.predict(start_date_str, end_date_str, utils.INPUT_DATA)

            # create entry new dataset
            dict_entry = {}
            for i, key in enumerate(utils.NPI_COLUMNS):
                dict_entry[key] = npi[i]
            tmp = self.raw_df[self.raw_df['GeoID'] == geo_id]['ConfirmedCases']
            for i, key in enumerate(utils.LB_DAYS_COLUMNS):
                dict_entry[key] = tmp.iloc[-(utils.NB_LOOKBACK_DAYS + i)]
            dict_entry['NewCases'] = df_pred['PredictedDailyNewCases'].values[-1]
            dataset.append(dict_entry)

        # create and save dataset dataframe
        df_dataset = pd.DataFrame(dataset, columns=utils.NPI_COLUMNS + utils.LB_DAYS_COLUMNS + ['NewCases'])
        df_dataset.to_csv(utils.SURROGATE_DATA_FILE, sep=',', index=False)

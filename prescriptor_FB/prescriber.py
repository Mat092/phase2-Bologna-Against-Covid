#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import utils
import datetime
import numpy as np
import pandas as pd
import docplex.mp.model as cpx

from sklearn import preprocessing
from keras.models import model_from_json

from eml.net.reader import keras_reader
from eml.backend import cplex_backend
from eml.net.embed import encode

from standard_predictor.xprize_predictor import XPrizePredictor


class PrescriberIP():
    """ Class representing the prescriber for a country"""
    def __init__(self, geo_id, net_prefix='nn'):
        self.geo_id = geo_id
        self.net_prefixes = net_prefix

        # Retrive Lookback days
        raw_df = utils.prepare_historical_df()
        self.scaler = preprocessing.MinMaxScaler()
        self.lb_days = raw_df[raw_df['GeoID'] == self.geo_id]['ConfirmedCases'].iloc[-utils.NB_LOOKBACK_DAYS:]
        self.lb_days = self.scaler.fit_transform(self.lb_days.values.reshape(-1, 1)).ravel()

        # Build a backend object and docplex model
        self.bkd = None
        self.mdl = None

        # Create elements optimization problem
        self.n_var = len(utils.NPI_COLUMNS + utils.LB_DAYS_COLUMNS)
        self.X_vars = None  # NN Input varibles
        self.Z_0_var = None # NN Output varaible
        self.Z_1_var = None # Stringency
        self.R_var = None   # Objective
        self.alpha = 0.7    # Trade-off objective

        # Create Input file for XPrizePredictor
        df_tmp = raw_df[raw_df['GeoID'] == self.geo_id]
        df_tmp.to_csv('tmp.csv', index=False)

    def load_keras_nets(self):
        # Load keras network
        with open(self.net_prefixes + '.json') as fp:
            self.knet = model_from_json(fp.read())
        # Load the model weights
        wgt_fname = os.path.join(self.net_prefixes + '.h5')
        self.knet.load_weights(wgt_fname)

    def convert_keras_net(self):
        # Convert Network into representation for EML
        self.net = keras_reader.read_keras_sequential(self.knet)

    def build_optimization_problem(self):
        self.bkd = cplex_backend.CplexBackend()
        self.mdl = cpx.Model()
        self.X_vars = []
        # Builds Variables of the optimization problem
        for i in range(self.n_var - len(self.lb_days)):
            # NPI input variables
            self.X_vars.append(self.mdl.continuous_var(lb=0, ub=1, name='in_' + str(i)))
        for i, lb_day in enumerate(self.lb_days):
            # Historical Lookback days
            self.X_vars.append(self.mdl.continuous_var(lb=lb_day, ub=lb_day, name='in_lb_' + str(i)))
        self.Z_0_var = self.mdl.continuous_var(lb=0, ub=1, name='out')
        self.Z_1_var = self.mdl.continuous_var(lb=0, ub=1, name='stringency')
        self.mdl.add_constraint(self.Z_1_var == utils.mean(self.X_vars[0:self.n_var - len(self.lb_days)]))
        self.R_var = self.mdl.continuous_var(lb=0, ub=np.inf, name='obj')
        self.mdl.add_constraint(self.R_var == self.alpha * self.Z_0_var + (1-self.alpha) * self.Z_1_var)
        self.mdl.set_objective('min', self.R_var)

    def encode(self):
        # Encodes the network into the optimization problem
        encode(self.bkd,
               self.net,
               self.mdl,
               self.X_vars,
               self.Z_0_var,
               'nn')

    def solve(self):
        # Solves the optimization problem
        print('=== Starting the solution process')
        ip = []
        start_date_str = "2021-1-1"
        for i in range(13):

            self.build_optimization_problem()
            encode(self.bkd, self.net, self.mdl, self.X_vars, self.Z_0_var, 'nn')

            # Solve Problem
            sol = self.mdl.solve()

            # Convert Solution into NPI
            sol = [sol[self.X_vars[i]] for i in range(self.n_var - len(self.lb_days))]
            npi = [int(x * utils.IP_MAX_VALUES[utils.NPI_COLUMNS[i]]) for i, x in enumerate(sol)]
            ip.append(npi)

            # Create New Input Data with prescribed NPI
            data = utils.create_input_data(self.geo_id, npi, start_date_str, 7)
            df_ips = pd.DataFrame(data)
            df_ips.to_csv(utils.INPUT_DATA, sep=',')

            # Predict Daily New Cases with New NPI
            predictor = XPrizePredictor(utils.MODEL_WEIGHTS_FILE, 'tmp.csv')
            end_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d') + datetime.timedelta(days=7)
            end_date_str = str(end_date).split(' ')[0]
            df_pred = predictor.predict(start_date_str, end_date_str, utils.INPUT_DATA)
            start_date_str = end_date_str
            print(df_pred)

            # Create Input New file for XPrizePredictor
            df_tmp = pd.read_csv('tmp.csv', delimiter=',')
            new_rows = df_tmp.iloc[-(len(df_pred) + 1):]
            # compute confirmed cases
            last_confirmed_cases = new_rows['ConfirmedCases'].iloc[0]
            new_rows = new_rows.iloc[1:]
            new_cases = np.array(df_pred['PredictedDailyNewCases'])
            new_confirmed_cases = []
            for i in range(len(new_rows)):
                new_confirmed_cases.append(sum(new_cases[0:i]) + last_confirmed_cases)
            new_rows['ConfirmedCases'] = new_confirmed_cases
            new_rows['NewCases'] = new_cases
            new_rows['Date'] = np.array(df_pred['Date'])
            df_tmp = df_tmp.append(new_rows)
            df_tmp.to_csv('tmp.csv', index=False)

            # Update Lookback Days
            new_confirmed_cases = self.scaler.fit_transform(np.array(new_confirmed_cases).reshape((-1, 1)))
            self.lb_days = np.append(self.lb_days[len(df_pred):], new_confirmed_cases)
        return ip

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import utils
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class SurrogateModel():
    """ Class Representing the surrogate model of the standard predictor for a country (& region)"""
    def __init__(self):
        # Build Dataset for Training
        self.df = pd.read_csv(utils.SURROGATE_DATA_FILE, delimiter=',')
        self.X = self.df[utils.NPI_COLUMNS + utils.LB_DAYS_COLUMNS]
        self.X = self.X.fillna(0)
        # Normalize NPI
        for npi_label in utils.NPI_COLUMNS:
            self.X[npi_label] = self.X[npi_label] / utils.IP_MAX_VALUES[npi_label]
        # Normalize Lookback days
        self.scaler = preprocessing.MinMaxScaler()
        for lb_label in utils.LB_DAYS_COLUMNS:
            self.X[lb_label] = self.scaler.fit_transform(self.X[lb_label].values.reshape(-1, 1))
        self.y = self.scaler.fit_transform(self.df['NewCases'].values.reshape(-1, 1))

    def build_model(self, batch_size=1, epochs=300, hlayers=4, width=8, actfun='linear'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.hlayers = hlayers
        self.width = width
        self.actfun = actfun
        # Split train-test set
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        # Stop criterion
        self.callback = EarlyStopping(monitor='loss', min_delta=1e-3, patience=20)
        # Model Creation
        self.model = Sequential()
        self.model.add(Dense(self.width,
                             activation='relu',
                             input_shape=(len(self.X.columns),)))
        for _ in range(self.hlayers):
            self.model.add(Dense(self.width,
                                 activation='relu'))
        self.model.add(Dense(1,
                             activation=self.actfun))
        # Set loss and optimizer
        self.model.compile(loss='mse', optimizer='adam')
        # Fit data
        self.model.fit(X_train,
                       y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=[self.callback])
        # self.model.fit(X_train,
        #                y_train,
        #                batch_size=self.batch_size,
        #                epochs=self.epochs)
        # Test model
        y_pred = self.model.predict(X_test)
        print('MAE: ' + str(np.mean(np.abs(y_pred.ravel() - y_test))))

    def save(self, filename='nn'):
        # Save model
        with open(filename + '.json', 'w') as fp:
            fp.write(self.model.to_json())
        self.model.save_weights(filename + '.h5')

# if __name__ == '__main__':
#
#     surrogate_model = SurrogateModel('Italy__')
#     surrogate_model.build_model()
#     surrogate_model.save()
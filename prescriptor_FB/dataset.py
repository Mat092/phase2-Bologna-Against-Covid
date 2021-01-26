#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import datetime

import pandas as pd
import numpy as np

from smt.sampling_methods import LHS
from standard_predictor.xprize_predictor import XPrizePredictor

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

DATA_FILE = "surrogate_dataset.csv"
MODEL_WEIGHTS_FILE = "standard_predictor/models/trained_model_weights.h5"


xlimits = np.array(list(zip([0] * len(IP_MAX_VALUES), [x + 1 for x in list(IP_MAX_VALUES.values())])))
sampling = LHS(xlimits=xlimits, criterion='center')
npi = list(map(int, (list(sampling(1)[0]))))

predictor = XPrizePredictor(MODEL_WEIGHTS_FILE, DATA_FILE)
preds_df = predictor.predict(start_date_str, next_day_str, DATA_FILE)

if __name__ == '__main__':

    print(create_surrogate_dataset('2021-01-01', '2021-01-23', 7, 100))
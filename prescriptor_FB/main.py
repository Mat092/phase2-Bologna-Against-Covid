#!/usr/bin/env python
# -*- coding: utf-8 -*-
import utils
import time
from dataset import DatasetGenerator
from model import SurrogateModel
from prescriber import PrescriberIP

if __name__ == '__main__':
    # Setting Parameters
    t0 = round(time.time())
    nsamples = 100
    lag_size = 7
    geo_id = 'Italy__'
    start_date_str = "2021-1-1"
    end_date_str = "2021-1-16"
    keys = utils.ID_COLS + utils.NPI_COLUMNS
    dt_generator = DatasetGenerator(nsamples, lag_size)
    dt_generator.generate_surrogate_dataset(geo_id, start_date_str)
    surrogate_model = SurrogateModel()
    surrogate_model.build_model()
    surrogate_model.save()
    precriber = PrescriberIP(geo_id)
    precriber.load_keras_nets()
    precriber.convert_keras_net()
    # precriber.encode()
    print(precriber.solve())
    t1 = round(time.time())
    print('Time:' + str((t1 - t0) / 60))


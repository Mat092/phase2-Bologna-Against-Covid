import numpy as np
import pandas as pd
import os
from utils import get_predictions, prepare_historical_df, CASES_COL, IP_COLS, IP_MAX_VALUES, add_geo_id, generate_costs,predict

NB_LOOKBACK_DAYS = 14
DATA_PATH = 'data'
HIST_DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')
TMP_PRED_FILE_NAME = 'preds.csv'
EVAL_START_DATE = '2020-03-01'
EVAL_END_DATE = '2020-03-10'
eval_start_date = pd.to_datetime(EVAL_START_DATE, format='%Y-%m-%d')
eval_end_date = pd.to_datetime(EVAL_END_DATE, format='%Y-%m-%d')


print("Loading historical data...")
df = prepare_historical_df()
df.head()
geos = df.GeoID.unique()
# Gather values for scaling network output
ip_max_values_arr = np.array([IP_MAX_VALUES[ip] for ip in IP_COLS])

cost_df = generate_costs(distribution='uniform')
cost_df = add_geo_id(cost_df)

pred_df = predict(EVAL_START_DATE,EVAL_END_DATE, HIST_DATA_FILE_PATH,TMP_PRED_FILE_NAME)
print(pred_df)

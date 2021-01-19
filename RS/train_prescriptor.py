import numpy as np
import pandas as pd

from utils import get_predictions, prepare_historical_df, CASES_COL, IP_COLS, IP_MAX_VALUES, add_geo_id, generate_costs

NB_LOOKBACK_DAYS = 14

EVAL_START_DATE = '2020-08-01'
EVAL_END_DATE = '2020-08-02'
eval_start_date = pd.to_datetime(EVAL_START_DATE, format='%Y-%m-%d')
eval_end_date = pd.to_datetime(EVAL_END_DATE, format='%Y-%m-%d')


print("Loading historical data...")
df = prepare_historical_df()

geos = df.GeoID.unique()
# Gather values for scaling network output
ip_max_values_arr = np.array([IP_MAX_VALUES[ip] for ip in IP_COLS])

cost_df = generate_costs(distribution='uniform')
cost_df = add_geo_id(cost_df)


# Make prescriptions one day at a time, feeding resulting
# predictions from the predictor back into the prescriptor.
for date in pd.date_range(eval_start_date, eval_end_date):
    date_str = date.strftime("%Y-%m-%d")

    # Prescribe for each geo
    geo_costs = {}
    past_cases = {}
    past_ips = {}
    for geo in geos:

        # Pull out historical data for all geos
        geo_df = df[df['GeoID'] == geo]
        past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))
        past_ips[geo] = np.array(geo_df[IP_COLS])

        costs = cost_df[cost_df['GeoID'] == geo]
        cost_arr = np.array(costs[IP_COLS])[0]
        geo_costs[geo] = cost_arr

        # Prepare input data. Here we use log to place cases
        # on a reasonable scale; many other approaches are possible.
        X_cases = np.log(past_cases[geo][-NB_LOOKBACK_DAYS:] + 1)
        X_ips = past_ips[geo][-NB_LOOKBACK_DAYS:]
        X_costs = geo_costs[geo]
        X = np.concatenate([X_cases.flatten(),
                            X_ips.flatten(),
                            X_costs])

       # TODO model fit

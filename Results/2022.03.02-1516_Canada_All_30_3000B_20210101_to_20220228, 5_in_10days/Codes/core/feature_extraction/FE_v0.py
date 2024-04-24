# from sklearn.metrics import accuracy_score, f1_score
# from sklearn import preprocessing
# import time
import os
# import json
# import pickle
# from random import choices
import gc
import time
import numpy as np
import pandas as pd
from core.feature_extraction.finta_me import FinTA
from talib.abstract import *
import multiprocessing as mp
from datetime import datetime, timedelta

# ))))))))))))))))))))))))))))))))))))))))))))))
# if configs['feature_extraction']['Concat_last_day']:
#     X_prev = X.iloc[:-1, 7:-2].reset_index(drop=True)
#     X_prev.columns = [f"prev_{x}" for x in X_prev.columns]
#     X = pd.concat([X.iloc[1:, :].reset_index(drop=True), X_prev], axis=1)
#     y = y[1:]
# ))))))))))))))))))))))))))))))))))))))))))))))


def load_preprocess_extract_features(data_paths, all_metadata, configs):
    st = time.time()
    do_multiprocessing = False
    if do_multiprocessing:
        # --- multiprocessing begins here ---
        n_process = configs.get("n_processes", 1)
        pool = mp.Pool(processes=n_process)

        results = [pool.apply_async(extract_features_for_each_ticker,
                                    args=(data_paths[i], all_metadata[i], configs))
                   for i in range(len(data_paths))]

        # results = [p for p in results]
        Xy = [p.get() for p in results]
        Xy = pd.concat(Xy)

        pool.close()
        pool.join()
        gc.collect()

    else:
        Xy = []
        metadata_list = all_metadata.reset_index().T.to_dict().values()
        for i, data_path, meta_data in zip(range(len(data_paths)), data_paths, metadata_list):
            print(f"Processing {(i+1)} / {len(data_paths)}: \t {meta_data['Symbol']}")
            x_y = extract_features_for_each_ticker(data_path, meta_data, configs)

            if i == 0:
                u, c = np.unique(x_y.columns, return_counts=True)
                print("Duplicated column names", u[c > 1], c[c > 1])
                assert len(u) == len(x_y.columns)

            if x_y is not None:
                Xy.append(x_y)
            else:
                print("Short history of ticker, Skip it !")

        Xy = pd.concat(Xy).reset_index(drop=True)

    print(f"Whole Dataset Generated with Shape of {Xy.shape}")

    # Drop Lookback window rows
    Xy = Xy[Xy['Date'] >= configs["data_selection"]["time_filtering"][0]]  #.iloc[lookback_w+1:, :]

    if configs["feature_extraction"]["Save_Generated_Dataset"]:
        sector = ' '.join(configs['data_selection']['ticker_filtering']['Sector'])
        m_caps = (configs['data_selection']['ticker_filtering']['MarketCap_in_Billion'])
        days = [x.replace('-', '') for x in configs['data_selection']['time_filtering']]
        sfn = f"GeneratedDatasets/{sector}_{days[0]}_to_{days[1]}_MarketCap={m_caps[0]}to{m_caps[1]}B.csv"
        Xy.to_csv(sfn, index=False)

    # Drop last rows where ground truth is not available yet
    # Xy = Xy.dropna(subset=['Label'])  #.reset_index(drop=True)

    for f in Xy.columns[:-3]:
        if Xy[f].isnull().values.any():
            print(f"{f} column .isnull().values.any()")
    # missings = Xy.iloc[:, :-3][Xy.iloc[:, :-3].isna().any(axis=1)]
    # print(missings[missings.columns[missings.isna().any(axis=0)]])
    # assert(not Xy.iloc[:, :-3].isnull().values.any())

    print("Feature extraction elapsed time (seconds) : ", time.time() - st)
    return Xy


def extract_features_for_each_ticker(data_path, ticker_meta_data, configs):

    # if configs['Verbose']:
    #     print(ticker_meta_data)

    # ---------------------------------------------- Load each data File ------------------------------------------
    data = pd.read_csv(data_path, parse_dates=['Date'])
    data = data.rename(columns={"Open": "open", 'Close': "close", "High": "high", "Low": "low", "Volume": "volume"}, )
    # data = data.reset_index()  # .drop('Date', axis=1)
    time_interval = configs["data_selection"]["time_filtering"]

    lookback_w = configs['data_selection']['lookback_window']
    if time_interval:
        # print("Clip time")
        temp_start_date = datetime.strptime(time_interval[0], "%Y-%m-%d") - timedelta(lookback_w)
        data = data[(data['Date'] > temp_start_date) & (data['Date'] <= time_interval[1])]
        data = data.reset_index(drop=True)

    if data.shape[0] < 50:
        return None
    file_name = os.path.split(data_path)[1]
    data['Ticker'] = file_name.split("_")[0]

    original_features = data[['Ticker', 'Date', 'close', 'open', 'high', 'low', 'volume']]
    Finta_features = extract_Finta_features(data, configs["feature_extraction"]['Finta_Indicators'])
    TAlib_features = extract_TALib_features(data, configs["feature_extraction"]['TALib_Indicators'])
    Fundamental_features = extract_fundamental_features(
        data, configs["feature_extraction"]['Fundamental_indicators'], ticker_meta_data)
    feature_set = pd.concat([original_features, Fundamental_features, Finta_features, TAlib_features], axis=1)

    # ---------------------------------------------- Define Targets (Labels) ----------------------------------------
    period = configs['Target_definition']["Period"]
    desired_increase = configs['Target_definition']["Increase_pct"]
    decrease_tolerance = configs['Target_definition']['decrease_tolerance']  # kind of stop loss!

    targets = calculate_ground_truth_target(data, period, desired_increase, decrease_tolerance)
    # --------------------------------------------------------------------------------------------------------------

    Xy = pd.concat([feature_set, targets], axis=1)  # [indicator[0]] = res1

    # print(f"feature_set.shape = {Xy.shape}")

    return Xy


def extract_Finta_features(data, feature_extractors):
    assert (data.shape[0] >= 50)
    feature_set = pd.DataFrame()

    for indicator in feature_extractors[:]:
        # if indicator[1] == "Max_increase_pct":
        #     print(indicator)
        if len(indicator) == 3:
            args = indicator[2]
            for arg in args:
                # arg = tuple([arg])
                this_indicator_features = eval(f"FinTA.{indicator[1]}(data, *{arg})")
                # this_indicator_features = eval(f"{indicator[1]}(data)")
                feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

        elif len(indicator) == 2:
            this_indicator_features = eval(f"FinTA.{indicator[1]}(data)")
            # this_indicator_features = eval(f"{indicator[1]}(data)")
            feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

    feature_set.columns = [f'FinTA.{x}' for x in feature_set.columns]

    return feature_set


def extract_TALib_features(data, feature_extractors):
    assert (data.shape[0] >= 50)
    feature_set = pd.DataFrame()

    for indicator in feature_extractors[:]:
        if len(indicator) == 3:
            args = indicator[2]
            for arg in args:
                this_indicator_features = eval(f"{indicator[1]}(data, *{arg})")
                if not (isinstance(this_indicator_features, pd.DataFrame)):
                    this_indicator_features = this_indicator_features.rename(f"{indicator[1]}_{arg.strip('(), ')}")
                if isinstance(this_indicator_features, pd.DataFrame):
                    this_indicator_features = this_indicator_features.add_suffix(f"_{arg.strip('(), ')}")
                feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

        elif len(indicator) == 2:
            this_indicator_features = eval(f"{indicator[1]}(data)")
            if not (isinstance(this_indicator_features, pd.DataFrame)):
                this_indicator_features = this_indicator_features.rename(indicator[1])
            feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

        # try:
        #     print(this_indicator_features.name)
        # except:
        #     print(this_indicator_features.columns)
    feature_set.columns = [f'TAlib.{x}' for x in feature_set.columns]

    return feature_set


def extract_fundamental_features(data, indicators, metadata):
    feature_set = data[['close']]
    # for indicator in indicators:
    n_shares = metadata['Market Cap'] / metadata['Last Sale']   # data['close'].iloc[-1]
    feature_set['Market Cap'] = np.round(n_shares * data['close'], 1)
    feature_set["Market_Cap_Fixed"] = np.round(metadata['Market Cap'], 1)

    feature_set.drop(['close'], axis=1, inplace=True)

    # feature_set.columns = [f'Fundamental.{x}' for x in feature_set.columns]

    return feature_set


def calculate_ground_truth_target(df, period, desired_increase, max_loss_tolerance):
    targets = []
    max_pct_increases = []
    max_pct_decreases = []
    for i in range(0, len(df) - period):
        target = 0
        curr_price = df['close'].iloc[i]
        max_price_in_the_next_n_candles = df['high'].iloc[i + 1:i + period + 1].max()
        id_max_price_in_the_next_n_candles = df['high'].iloc[i + 1:i + period + 1].idxmax()
        max_pct_increase = (max_price_in_the_next_n_candles - curr_price) / curr_price * 100

        # max loss -- > we use the percentage loss before reaching the target
        if max_pct_increase > desired_increase:
            min_price_from_current_until_max_price = df['low'].iloc[i:id_max_price_in_the_next_n_candles].min()
            # min_closing_price_since_current_until_reaching_max_price = data['close'].iloc[i:id_max_price_in_the_next_n_candles].min()
            max_pct_decrease = (min_price_from_current_until_max_price - curr_price) / curr_price * 100
        else:  # if it didn't increase by desired_increase, the loss is
            min_price_at_the_end_of_days_in_period = df['close'].iloc[i+1:i+period+1].min()
            max_pct_decrease = (min_price_at_the_end_of_days_in_period - curr_price) / curr_price * 100
            max_pct_decrease = max(-3, max_pct_decrease)

        if (max_pct_increase > desired_increase) & (max_pct_decrease > max_loss_tolerance):
            target = 1
        # if (max_pct_increase > desired_increase-1) & (max_pct_decrease > decrease_tolerance):
        #     target_1 = 1
        # if (max_pct_increase > desired_increase-2) & (max_pct_decrease > decrease_tolerance):
        #     target_2 = 1
        targets.append(target)
        max_pct_increases.append(max_pct_increase)
        max_pct_decreases.append(max_pct_decrease)

    targets.extend([np.nan] * period)
    max_pct_increases.extend([np.nan] * period)
    max_pct_decreases.extend([np.nan] * period)

    targets = pd.DataFrame({'Label': targets, 'Max_increase': max_pct_increases,
                            'Max_decrease': max_pct_decreases})  # index=df['Date']

    return targets

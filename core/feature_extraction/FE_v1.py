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
            print(f"Processing {(i+1)} / {len(data_paths)} file ...")
            x_y = extract_features_for_each_ticker(data_path, meta_data, configs)
            if x_y is not None:
                Xy.append(x_y)
            else:
                print("Short history of ticker, Skip it !")

        Xy = pd.concat(Xy).reset_index(drop=True)

    print(f"Whole Dataset Generated with Shape of {Xy.shape}")

    if configs["feature_extraction"]["Save_Generated_Dataset"]:
        sector = ' '.join(configs['data_selection']['ticker_filtering']['Sector'])
        m_caps = (configs['data_selection']['ticker_filtering']['MarketCap_in_Billion'])
        days = [x.replace('-', '') for x in configs['data_selection']['time_filtering']]
        sfn = f"GeneratedDatasets/{sector}_{days[0]}_to_{days[1]}_MarketCap={m_caps[0]}to{m_caps[1]}B.csv"
        Xy.to_csv(sfn, index=False)

    print("Feature extraction elapsed time (seconds) : ", time.time() - st)
    return Xy  # , samples_meta_data_per_key


def extract_features_for_each_ticker(data_path, ticker_meta_data, configs):

    # if configs['Verbose']:
    #     print(ticker_meta_data)

    # ---------------------------------------------- Load each data File ------------------------------------------
    data = pd.read_csv(data_path, parse_dates=['Date'])
    data = data.rename(columns={"Open": "open", 'Close': "close", "High": "high", "Low": "low", "Volume": "volume"}, )
    # data = data.reset_index()  # .drop('Date', axis=1)
    time_interval = configs["data_selection"]["time_filtering"]

    # lookback_w = configs['data_selection']['lookback_window']
    # if time_interval:
    #     # print("Clip time")
    #     temp_start_date = datetime.strptime(time_interval[0], "%Y-%m-%d") - timedelta(lookback_w)
    #     data = data[(data['Date'] > temp_start_date) & (data['Date'] <= time_interval[1])]
    #     data = data.reset_index(drop=True)

    if data.shape[0] < 50:
        return None
    file_name = os.path.split(data_path)[1]
    data['Ticker'] = file_name.split("_")[0]

    original_features = data[['Ticker', 'Date', 'close', 'open', 'high', 'low', 'volume']]
    original_features = original_features[original_features['Date'] >= time_interval[0]].set_index('Date')
    Finta_features = extract_Finta_features(data, configs["feature_extraction"]['Finta_Indicators'], time_interval)
    TAlib_features = extract_TALib_features_v2(data, configs["feature_extraction"]['TALib_Indicators'], time_interval)
    Fundamental_features = extract_fundamental_features(
        data, configs["feature_extraction"]['Fundamental_indicators'], ticker_meta_data, time_interval[0])
    feature_set = pd.concat([original_features, Fundamental_features, Finta_features, TAlib_features], axis=1)

    # ---------------------------------------------- Define Targets (Labels) ----------------------------------------
    period = configs['Target_definition']["Period"]
    desired_increase = configs['Target_definition']["Increase_pct"]
    decrease_tolerance = configs['Target_definition']['decrease_tolerance']  # kind of stop loss!

    targets = calculate_ground_truth_target(original_features.reset_index(), period, desired_increase, decrease_tolerance)

    Xy = pd.concat([feature_set, targets], axis=1)  # [indicator[0]] = res1

    Xy = Xy.reset_index()
    Xy = Xy[Xy['Date'] >= time_interval[0]]   #.iloc[lookback_w+1:, :]
    print(f"feature_set.shape = {Xy.shape}")

    return Xy


def extract_Finta_features(data, feature_extractors, time_interval):
    assert (data.shape[0] >= 50)
    real_start_ind = np.where(data['Date'] == time_interval[0])[0][0]
    feature_set = pd.DataFrame({'Date': data['Date'].loc[real_start_ind:].reset_index(drop=True)})

    for indicator in feature_extractors[:]:
        # print(f"\t {indicator}")

        if len(indicator) == 3:
            args = indicator[2]
            for arg in args:
                # arg = tuple([arg])
                # this_indicator_features = eval(f"FinTA.{indicator[1]}(data, *{arg})")
                # feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

                arg_val = min(400, 2 * max([x for x in eval(arg)]))
                temp_start_ind = real_start_ind - arg_val
                temp_data = data[temp_start_ind:]
                this_indicator_features = eval(f"FinTA.{indicator[1]}(temp_data, *{arg})")
                # temp = data.loc[real_start_ind:].join(this_indicator_features.loc[real_start_ind:])
                # print(len(temp))
                this_indicator_features = this_indicator_features.loc[real_start_ind:].reset_index(drop=True)  # drop=True
                if this_indicator_features.isnull().values.any():
                    print("df.isnull().values.any()")
                assert(len(temp_data) == len(this_indicator_features) + arg_val)
                feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

                # time_interval = configs["data_selection"]["time_filtering"]
                # lookback_w = arg  # configs['data_selection']['lookback_window']
                #     temp_start_date = datetime.strptime(time_interval[0], "%Y-%m-%d") - timedelta(lookback_w)
                #     data = data[(data['Date'] > temp_start_date) & (data['Date'] <= time_interval[1])]
                #     data = data.reset_index(drop=True)

        elif len(indicator) == 2:
            print("len(indicator) = 2")
            temp_start_ind = real_start_ind - 40
            temp_data = data[temp_start_ind:]
            this_indicator_features = eval(f"FinTA.{indicator[1]}(temp_data)")
            this_indicator_features = this_indicator_features.loc[real_start_ind:].reset_index(drop=True)
            if this_indicator_features.isnull().values.any():
                print("df.isnull().values.any()")

            assert (len(temp_data) == len(this_indicator_features) + 40)
            feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

            # this_indicator_features = eval(f"FinTA.{indicator[1]}(data)")
            # feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

    feature_set = feature_set.set_index('Date')
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
                    this_indicator_features = this_indicator_features.rename(indicator[1])
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


def extract_TALib_features_v2(data, feature_extractors, time_interval):
    assert (data.shape[0] >= 50)
    real_start_ind = np.where(data['Date'] == time_interval[0])[0][0]
    feature_set = pd.DataFrame({'Date': data['Date'].loc[real_start_ind:].reset_index(drop=True)})

    for indicator in feature_extractors[:]:
        if len(indicator) == 3:
            args = indicator[2]
            for arg in args:

                arg_val = min(400, 2 * max([x for x in eval(arg)]))
                temp_start_ind = real_start_ind - arg_val
                temp_data = data[temp_start_ind:]
                this_indicator_features = eval(f"{indicator[1]}(temp_data, *{arg})")
                # temp = data.loc[real_start_ind:].join(this_indicator_features.loc[real_start_ind:])
                # print(len(temp))
                this_indicator_features = this_indicator_features.loc[real_start_ind:].reset_index(drop=True)  # drop=True
                if this_indicator_features.isnull().values.any():
                    print(f"{indicator}: df.isnull().values.any()")
                assert(len(temp_data) == len(this_indicator_features) + arg_val)

                if not (isinstance(this_indicator_features, pd.DataFrame)):
                    this_indicator_features = this_indicator_features.rename(indicator[1])
                feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

        elif len(indicator) == 2:
            temp_start_ind = real_start_ind - 100
            temp_data = data[temp_start_ind:]
            this_indicator_features = eval(f"{indicator[1]}(temp_data)")
            this_indicator_features = this_indicator_features.loc[real_start_ind:].reset_index(drop=True)
            if this_indicator_features.isnull().values.any():
                print(f"{indicator}: df.isnull().values.any()")
            assert (len(temp_data) == len(this_indicator_features) + 100)

            if not (isinstance(this_indicator_features, pd.DataFrame)):
                this_indicator_features = this_indicator_features.rename(indicator[1])

            feature_set = pd.concat([feature_set, this_indicator_features], axis=1)

    feature_set = feature_set.set_index('Date')
    feature_set.columns = [f'TAlib.{x}' for x in feature_set.columns]

    return feature_set


def extract_fundamental_features(data, indicators, metadata, start_date):
    feature_set = data[['Date', 'close']]
    for indicator in indicators:
        feature_set[indicator] = np.round(metadata[indicator] * data['close'] / data['close'].iloc[-1], 1)
    feature_set = feature_set[feature_set['Date'] >= start_date]  #.iloc[lookback_w+1:, :]

    feature_set.drop(['close'], axis=1, inplace=True)
    feature_set = feature_set.set_index('Date')
    feature_set.columns = [f'Fundamental.{x}' for x in feature_set.columns]

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
        min_price_from_current_until_max_price = df['low'].iloc[i:id_max_price_in_the_next_n_candles].min()
        # min_closing_price_since_current_until_reaching_max_price = data['close'].iloc[i:id_max_price_in_the_next_n_candles].min()
        max_pct_decrease = (min_price_from_current_until_max_price - curr_price) / curr_price * 100

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
                            'Max_decrease': max_pct_decreases}, index=df['Date'])

    return targets

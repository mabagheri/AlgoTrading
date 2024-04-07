import json
import time
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
# import pandas as pd
from core.initialization.load_data_file_paths import load_selected_data_files_paths
# from core.classification.learning_utils import encode_categorical_labels_to_numerical
# from core.classification.main import classify
from core.feature_extraction.FE_v0 import load_preprocess_extract_features
import warnings
warnings.filterwarnings("ignore")
np.random.seed(135)
random.seed(135)


def get_predictions(model_folder, start_day, end_day):
    configs = json.load(open('Results/' + model_folder + '/configs.json', 'r'))
    configs["data_selection"]["time_filtering"] = [start_day, end_day]
    performances = pd.read_csv("Results/" + model_folder + '/performances.csv')
    precision_1_test = performances.iloc[1, 4]
    # 1:
    ""
    data_paths, metadata = load_selected_data_files_paths(configs)

    # 2: Get Xy for prediction (future)
    Xy = load_preprocess_extract_features(data_paths, metadata, configs)
    print("Xy.shape", Xy.shape)

    samples_for_prediction = Xy[Xy['Label'].isnull()]
    y_samples_for_prediction = samples_for_prediction['Label']
    cols_not_for_training = ["Ticker", 'Date', 'close', 'open', 'high', 'low', 'volume', 'Label', 'Max_increase', 'Max_decrease']
    X_samples_for_prediction = samples_for_prediction.drop(cols_not_for_training, axis=1)

    print("X_samples_for_prediction", X_samples_for_prediction.shape)

    # 3:
    model = xgb.Booster()  # init model
    model.load_model('Results/' + model_folder + '/model_sklearn.json')  # load data

    d_test = xgb.DMatrix(X_samples_for_prediction, label=y_samples_for_prediction, nthread=2)

    prob_y_test = model.predict(d_test)
    result_fold_ts = samples_for_prediction[['Ticker', 'Date']]
    result_fold_ts[model_folder] = prob_y_test

    # n =2
    # import shap
    # import matplotlib.pyplot as plt
    # top_n_predictions = result_fold_ts[model_folder].nlargest(n).reset_index()['index'].values
    # X_top_n = X_samples_for_prediction.loc[top_n_predictions]
    # for x in top_n_predictions:
    #     explainer = shap.Explainer(model)
    #     shap_values = explainer(X_top_n)
    #     shap.plots.force(shap_values[0])
    #     plt.show()

    # configs = json.load(open(model_folder + '/configs.json', 'r'))
    # first_day = samples_for_prediction['Date'].min().strftime("%Y%m%d")
    # last_day = samples_for_prediction['Date'].max().strftime("%Y%m%d")
    # Market = str(configs['data_selection']['ticker_filtering']['Country'][0])
    # m_caps = configs['data_selection']['ticker_filtering']['MarketCap_in_Billion']

    # result_fold_ts.to_csv(f'Results/Future_prediction/{first_day}_to_{last_day}_{Market}_{m_caps[0]}_{m_caps[1]}B.csv', index=False)

    # result_fold_ts = pd.concat([X_test[cols_not_for_training].reset_index(drop=True),
    #                             result_fold_ts.reset_index(drop=True)], axis=1)
    return pd.DataFrame(result_fold_ts), precision_1_test


st = time.time()

today = datetime.datetime.today().strftime('%Y-%m-%d')
end_date = today
start_date = str(datetime.date.today() + datetime.timedelta(days=-3))  # "2022-02-03"
print(start_date, end_date)

models = [
    ['Jan2022/2022.02.01-1153_All_10_3000B_20210101_to_20220131, 5_in_5days', start_date, end_date],
    ['Jan2022/2022.02.01-1203_All_30_3000B_20210101_to_20220131, 5_in_5days', start_date, end_date],
    ['Jan2022/+2022.01.30-2245_All_40_3000B_20210101_to_20220128, 7_in_14days', start_date, end_date],
    ['Jan2022/+2022.01.30-2356_All_40_3000B_20210101_to_20220128, 5_in_10days', start_date, end_date],
    ['Jan2022/+2022.01.31-2246_All_10_3000B_20210101_to_20220131, 5_in_10days', start_date, end_date],
    ['Jan2022/+2022.01.31-2256_All_10_3000B_20210501_to_20211231, 7_in_14days',  start_date, end_date],
    # ['Jan2022/2022.01.31-2305_All_10_3000B_20210101_to_20220131, 7_in_14days',  start_date, end_date],
    # ['2022.03.02-1516_Canada_All_30_3000B_20210101_to_20220228, 5_in_10days',  start_date, end_date],
    # ['2022.03.02-1514_Canada_All_30_3000B_20210101_to_20220228, 4_in_4days',  start_date, end_date],
    # ['2022.03.02-1505_Canada_All_5_3000B_20210101_to_20220228, 4_in_4days',  start_date, end_date],
    # ['2022.03.02-1500_Canada_All_5_3000B_20210101_to_20220228, 7_in_14days',  start_date, end_date],
    # ['',  start_date, end_date],
    ]

res, Precision_1_test = get_predictions(models[0][0], models[0][1], models[0][2])
classifiers_weights = [Precision_1_test]

for model_folder1, from_date, to_date in models[1:]:
    print(f"\n {model_folder1}")
    res1, Precision_1_test = get_predictions(model_folder1, from_date, to_date)
    print(res1)
    res = pd.merge(res, res1, how='outer', on=['Ticker', 'Date'])
    classifiers_weights.append(Precision_1_test)

print(res)
res['Weighted Avg'] = np.average(res.iloc[:, 2:2+len(models)], axis=1, weights=classifiers_weights)
res['Mean'] = np.mean(res.iloc[:, 2:2+len(models)], axis=1)
res['Prod'] = np.prod(res.iloc[:, 2:2+len(models)], axis=1)

now = time.strftime('%Y.%m.%d-%H%M')
res.to_csv(f'Results/+ Future_prediction/{now}.csv', index=False)

print(classifiers_weights)
print("Elapsed time (seconds) : ", (time.time() - st)/60)
import json
import random
import numpy as np
import xgboost as xgb
import pandas as pd
# from core.initialization.load_data_file_paths import load_selected_data_files_paths
# from core.classification.learning_utils import encode_categorical_labels_to_numerical
# from core.classification.main import classify
# from core.feature_extraction.FE_v0 import load_preprocess_extract_features
import warnings

warnings.filterwarnings("ignore")
np.random.seed(135)
random.seed(135)

# folder = r'Results\2021.09.24\All_10_100B_20200601_to_20210918, AUC=0.725, Pr=0.764'
# folder = r'Results\2021.09.24\All_100_3000B_20200601_to_20210918, AUC=0.723, Pr=0.516'
folder = r'Results/2021.11.01-1308_All_100_3000B_20200601_to_20211031, 10_in_15days'
configs = json.load(open(folder + '/configs.json', 'r'))

# 1: Read data
sector = ' '.join(configs['data_selection']['ticker_filtering']['Sector'])
m_caps = (configs['data_selection']['ticker_filtering']['MarketCap_in_Billion'])
days = [x.replace('-', '') for x in configs['data_selection']['time_filtering']]
sfn = f"GeneratedDatasets/{sector}_{days[0]}_to_{days[1]}_MarketCap={m_caps[0]}to{m_caps[1]}B.csv"
Xy = pd.read_csv(sfn)

# 2: Get Xy for prediction (future)
samples_for_prediction = Xy[Xy['Label'].isnull()]
y_samples_for_prediction = samples_for_prediction['Label']
cols_not_for_training = ["Ticker", 'Date', 'close', 'open', 'high', 'low', 'volume', 'Label', 'Max_increase', 'Max_decrease']
X_samples_for_prediction = samples_for_prediction.drop(cols_not_for_training, axis=1)

# 3:
model = xgb.Booster()  # init model
configs = json.load(open(folder + '/configs.json', 'r'))
model.load_model(folder + '/model_sklearn.json')  # load data

d_test = xgb.DMatrix(X_samples_for_prediction, label=y_samples_for_prediction, nthread=2)

prob_y_test = model.predict(d_test)
result_fold_ts = samples_for_prediction[['Ticker', 'Date']]
result_fold_ts['y_pred'] = prob_y_test

first_day = samples_for_prediction['Date'].min()
last_day = samples_for_prediction['Date'].max()
result_fold_ts.to_csv(f'Results/Future_prediction/{first_day}_to_{last_day}_{sector}_{m_caps[0]}_{m_caps[1]}B.csv', index=False)

# result_fold_ts = pd.concat([X_test[cols_not_for_training].reset_index(drop=True),
#                             result_fold_ts.reset_index(drop=True)], axis=1)

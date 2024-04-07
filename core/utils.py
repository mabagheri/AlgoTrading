import numpy as np
import pandas as pd


def missing_percentage(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    results = pd.DataFrame({"Columns": df.columns,
                            'Missing_percent': percent_missing})
    return results


def get_list_of_files_in_a_folder(folder, extension='.csv'):
    import os
    files_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                files_list.append(os.path.join(root, file))

    return files_list


def get_top_correlations_v2(corr_vals):
    keep = np.triu(np.ones(corr_vals.shape), k=1).astype('bool').reshape(corr_vals.size)
    res = corr_vals.unstack()[keep].reset_index()
    res = pd.DataFrame(res).rename(columns={'level_0': 'feature 1', 'level_1': 'feature 2', 0: 'Correlation'})

    return res.sort_values(by='Correlation', ascending=False).reset_index(drop=True)


def get_top_abs_correlations_v2(corr_vals):
    keep = np.triu(np.ones(corr_vals.shape), k=1).astype('bool').reshape(corr_vals.size)
    res = corr_vals.abs().unstack()[keep].reset_index()
    res = pd.DataFrame(res).rename(columns={'level_0': 'feature 1', 'level_1': 'feature 2', 0: 'Correlation'})

    return res.sort_values(by='Correlation', ascending=False).reset_index(drop=True)

# def scaling(X_train, X_test, style):
#
#     scaler = []
#     if style == "MinMax":
#         if X_train.ndim < 3:
#             min_max_scalar = preprocessing.MinMaxScaler()
#             X_train_scaled = min_max_scalar.fit_transform(X_train)
#             X_test_scaled = min_max_scalar.transform(X_test)
#
#         elif X_train.ndim == 3:
#             X_train_scaled = np.zeros(X_train.shape)
#             X_test_scaled = np.zeros(X_test.shape)
#             scalers = {}
#             for i in range(X_train.shape[-1]):
#                 scalers[i] = preprocessing.MinMaxScaler()
#                 X_train_scaled[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])
#                 X_test_scaled[:, :, i] = scalers[i].transform(X_test[:, :, i])
#
#         elif X_train.ndim == 4:
#             X_train_scaled = np.zeros(X_train.shape)
#             X_test_scaled = np.zeros(X_test.shape)
#             scalers = {}
#
#             original_shape = X_train.shape
#             original_shape_test = X_test.shape
#
#             X_train_reshaped = X_train.reshape((original_shape[0], original_shape[1] * original_shape[2],
#                                                 original_shape[3]))
#             X_test_reshaped = X_test.reshape((original_shape_test[0], original_shape_test[1] * original_shape_test[2],
#                                               original_shape_test[3]))
#             for i in range(X_train.shape[-1]):
#                 scalers[i] = preprocessing.MinMaxScaler()
#
#                 X_train_scaled[:, :, :, i] = scalers[i].fit_transform(X_train_reshaped[:, :, i]).reshape(original_shape[:-1])
#
#                 X_test_scaled[:, :, :, i] = scalers[i].transform(X_test_reshaped[:, :, i]).reshape(original_shape_test[:-1])
#
#     elif style == "StandardScaler":  # transform the features such that each feature has mean of 0 and std of 1
#         scaler = preprocessing.StandardScaler().fit(X_train)
#         X_train_scaled = scaler.transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#
#     elif style == "StandardScaler_2D":
#         X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
#         X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
#
#         scaler = preprocessing.StandardScaler().fit(X_train_reshaped)
#         X_train_scaled = scaler.transform(X_train_reshaped)
#         X_test_scaled = scaler.transform(X_test_reshaped)
#
#         X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
#         X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
#
#     elif style == "MinMax_2D":
#         X_train_scaled = np.zeros(X_train.shape)
#         X_test_scaled = np.zeros(X_test.shape)
#
#         for i in range(X_train.shape[-1]):
#             this_min = np.min(X_train[:, :, i])
#             this_max = np.max(X_train[:, :, i])
#
#             X_train_scaled[:, :, i] = (X_train[:, :, i] - this_min) / this_max
#             X_test_scaled[:, :, i]  = (X_test[:, :, i] - this_min) / this_max
#
#     elif style == "None":
#         X_train_scaled = X_train
#         X_test_scaled = X_test
#
#     elif style == "SimpleReduction":
#         X_train_scaled = X_train / 50
#         X_test_scaled = X_test / 50
#
#     return X_train_scaled, X_test_scaled, scaler

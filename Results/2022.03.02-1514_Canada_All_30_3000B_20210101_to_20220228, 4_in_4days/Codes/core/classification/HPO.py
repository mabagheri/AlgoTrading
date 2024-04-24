"""
Main file for HyperParameter Optimization
"""
import pandas as pd
from sklearn.model_selection import ParameterGrid
from core.classification.classify_v2 import classify


def Perform_HPO_GridSearch(configs, Xy):
    configs["save_the_results"]["save_json_file"] = False

    saving_folder = configs['saving_folder']

    # Run HPO Grid Search
    exp_n = 1
    tr_results = []
    ts_results = []

    # param_grid = {'max_depth': [3, 4], 'n_trees': [20], 'eta': [0.05]}
    param_grid = {'max_depth': [3, 4, 5], 'n_trees': [50, 100, 250], 'eta': [0.03]}
    for i, pars in enumerate(list(ParameterGrid(param_grid))):
        print(f"\n Experiment: {exp_n} / {len(ParameterGrid(param_grid))}")
        configs['classification']["XGBoost: max_depth"] = pars['max_depth']
        configs['classification']["XGBoost: n_trees"] = pars['n_trees']
        configs['classification']["XGBoost: learning_rate"] = pars['eta']
        classification_results = classify(Xy, configs)
        # all_results.append(classification_results)
        results_cols = list(classification_results.columns)
        tr_results.append([pars['max_depth'], pars['n_trees'], pars['eta']] + list(
            classification_results.loc['Train', results_cols]))
        ts_results.append([pars['max_depth'], pars['n_trees'], pars['eta']] + list(
            classification_results.loc['Test', results_cols]))

        df_train = pd.DataFrame(tr_results, columns=["max_depth", "n_trees", "eta"] + results_cols)
        df_test = pd.DataFrame(ts_results, columns=["max_depth", "n_trees", "eta"] + results_cols)

        exp_n += 1
        with pd.ExcelWriter(f"{saving_folder}GridSearch_Results.xlsx") as writer:
            df_train.to_excel(writer, sheet_name='Train')
            df_test.to_excel(writer, sheet_name='Test')

    metric = 'Precision'
    best_idx = df_test[['Precision@0.5%', 'Precision@1%', 'Precision@3%', 'Precision@5%']].mean(axis=1).idxmax()
    configs['classification']["XGBoost: max_depth"] = int(df_test.loc[best_idx]['max_depth'])
    configs['classification']["XGBoost: n_trees"] = int(df_test.loc[best_idx]['n_trees'])
    configs['classification']["XGBoost: learning_rate"] = float(df_test.loc[best_idx]['eta'])

    print(f"\nBest performance with {metric} on the test dara obtained at #Exp{best_idx}, "
          f" max_depth={configs['classification']['XGBoost: max_depth']}"
          f" n_trees={configs['classification']['XGBoost: n_trees']}"
          f" alpha={configs['classification']['XGBoost: learning_rate']}")

    configs["save_the_results"]["save_json_file"] = True

    return configs

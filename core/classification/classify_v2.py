import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from core.classification import classify_by_XGBoost_v3  #, classify_by_FCN
from core.classification.evaluate_v2 import evaluate_clf_performance
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)


def classify(Xy, configs):

    # Dummy columns (those that will not be used for training)
    cols_not_for_training = ["Ticker", 'Date', 'close', 'open', 'high', 'low', 'volume', 'Max_increase', 'Max_decrease']

    # ---------------------------------------- 1. Get X and y  ----------------------------------------------------
    X_without_ground_truth = Xy[Xy['Label'].isna()].reset_index(drop=True).drop('Label', axis=1)
    # assert(not X_without_ground_truth.iloc[:, :-2].isnull().values.any())

    Xy = Xy.dropna(subset=['Label']).reset_index(drop=True)
    y = Xy['Label'].astype(int)
    X = Xy.drop('Label', axis=1)

    # -------  2. split data into train and test and train and test a model in each time (each fold) ---------------
    OOT_split_date = configs['validation']["OOT_split_date"]
    period = configs['Target_definition']['Period']
    OOT_split_date_test = datetime.strptime(OOT_split_date, "%Y-%m-%d") + timedelta(period+2)

    train_ind = np.where(X['Date'] <= OOT_split_date)[0]
    test_ind = np.where(X['Date'] > OOT_split_date_test)[0]

    X_train = X.iloc[train_ind, :]
    y_tr = y[train_ind]
    X_test = X.iloc[test_ind, :]
    y_ts = y[test_ind]

    # ----------------------------- 3. Classification  begins here -------------------------------------------
    classifier = configs["classification"]["classifier"]
    model = prob_y_tr = prob_y_ts = prob_y_without_ground_truth = None

    #
    if classifier == "XGBoost":
        model, prob_y_tr, prob_y_ts, prob_y_without_ground_truth = classify_by_XGBoost_v3.classify(
            X_train.drop(cols_not_for_training, axis=1), y_tr,
            X_test.drop(cols_not_for_training, axis=1), y_ts,
            X_without_ground_truth.drop(cols_not_for_training, axis=1, errors='ignore'), configs)

    # model.predict(xgb.DMatrix(X_train, label=y_train, nthread=2)Xy_without_ground_truth)
    # if classifier == "FCN":
    #     model, prob_y_tr, prob_y_ts = classify_by_FCN.classify(
    #         X_train.drop(cols_not_for_training, axis=1), y_tr,
    #         X_test.drop(cols_not_for_training, axis=1), y_ts, configs)

    result_fold_tr = pd.DataFrame({'y_true': y_tr, 'y_pred': prob_y_tr})
    # also add to above 'Max Increase': X_train['Max_increase'], 'Max_Decrease': X_train['Max_Decrease']})
    result_fold_ts = pd.DataFrame({'y_true': y_ts, 'y_pred': prob_y_ts})
    result_fold_ts = pd.concat([X_test[cols_not_for_training].reset_index(drop=True),
                                result_fold_ts.reset_index(drop=True)], axis=1)

    pred_X_without_y = pd.DataFrame({'y_pred': prob_y_without_ground_truth})
    pred_X_without_y = pd.concat([X_without_ground_truth[cols_not_for_training].reset_index(drop=True),
                                  pred_X_without_y.reset_index(drop=True)], axis=1)

    # ---------- 4: Calculating Classification Performance Metrics and save ---------------------------------------------
    print("Calculating Classification Performance Metrics ...")
    gr_truth_and_prediction_train = result_fold_tr
    gr_truth_and_prediction_test = result_fold_ts
    results_tr = evaluate_clf_performance(gr_truth_and_prediction_train,
                                          metrics=['AUPRC', 'Precision',  'CM'],  # 'Recall', 'Lift', 'AUC', 'Accuracy'
                                          top_n_percentiles=[0.5, 1, 3, 5, 10])
    results_ts = evaluate_clf_performance(gr_truth_and_prediction_test,
                                          metrics=['AUPRC', 'Precision', 'CM'],  # 'Recall', 'Lift', 'AUC', 'Accuracy'
                                          top_n_percentiles=[0.5, 1, 3, 5, 10])
    results = pd.DataFrame([results_tr, results_ts], index=['Train', 'Test'])

    if configs["save_the_results"]["save_json_file"]:
        results = save_outputs(results_tr, results_ts, gr_truth_and_prediction_train, gr_truth_and_prediction_test,
                               pred_X_without_y, X_train, model, cols_not_for_training, configs)

    return results


def save_outputs(result_tr, result_ts, gr_truth_and_prediction_tr, gr_truth_and_prediction_ts,
                 pred_X_without_y, X_train, learner, cols_not_for_training, config):
    results = pd.DataFrame([result_tr, result_ts], index=['Train', 'Test'])

    # ------------------------------------- Save Results, Confusion Matrix and Model ---------------------------------
    saving_folder = config['saving_folder']
    results.to_csv(saving_folder + 'performances.csv')
    results_filename = saving_folder + "configs.json"

    sector = ' '.join(config['data_selection']['ticker_filtering']['Sector'])
    m_caps = (config['data_selection']['ticker_filtering']['MarketCap_in_Billion'])
    inc = config['Target_definition']['Increase_pct']
    period = config['Target_definition']['Period']
    dec_tol = config['Target_definition']['decrease_tolerance']
    days = [x.replace('-', '') for x in config['data_selection']['time_filtering']]
    HPO = config['classification'].get('perform_HPO_GridSearch', False)
    xgb_depth = config['classification']["XGBoost: max_depth"]
    xgb_n_trees = config['classification']["XGBoost: n_trees"]

    test_meta_data_and_results = {
        'Market': config['data_selection']['ticker_filtering']['Country'][0],
        'Sector': sector, 'Market Caps': m_caps, "From": days[0], "To": days[1],
        "OOT split date": config['validation']["OOT_split_date"],
        "Increase": inc, "Period": period, "Dec. Tol.": dec_tol, 'HPO': HPO,
        "XGB_depth": xgb_depth, "XGB_n_trees": xgb_n_trees,
        "Folder": saving_folder
    }
    test_meta_data_and_results.update(result_ts)
    results_before = pd.read_csv("Results/All_Results.csv")
    results_this_exp = pd.DataFrame([test_meta_data_and_results], index=['Test'])
    all_results = pd.concat([results_before, results_this_exp], ignore_index=True)
    all_results.to_csv('Results/All_Results.csv', index=False)
    with open(results_filename, 'w') as outfile:
        json.dump(config, outfile, separators=(',', ':'), indent=4)

    pred_X_without_y.to_csv(saving_folder + 'pred_X_without_y.csv', index=False)

    if config["save_the_results"]["save_post_analysis"] or config["save_the_results"]["save_detailed_results"]:
        gr_truth_and_prediction_ts.to_csv(saving_folder + 'test_detailed_result.csv', index=False)
        gr_truth_and_prediction_tr.to_csv(saving_folder + 'train_detailed_result.csv', index=False)

    if config["save_the_results"]["save_model"]:
        learner.save_model(saving_folder + "model_sklearn.json")  # save model in JSON format
        # model.save_model("model_sklearn.txt")                 # save model in text format

    if config["save_the_results"]["save_shap_feat_importance"]:
        print('Generating shap values')
        import shap
        st = time.time()
        explainer = shap.Explainer(learner)
        shap_values = explainer(X_train.drop(cols_not_for_training, axis=1))
        shap_feat_importance = pd.DataFrame(np.abs(shap_values.values).mean(axis=0),
                                            index=shap_values.feature_names, columns=['feat_importance'])
        shap_feat_importance.to_csv(saving_folder + "shap_feat_importance.csv", index=True)
        # sort_values(by='feat_importance', ascending=False).
        # pd.DataFrame(shap_values.values, columns=shap_values.feature_names).to_csv(saving_folder + "shap_values.csv", index=False)
        # pd.DataFrame(shap_values.data, columns=shap_values.feature_names).to_csv(saving_folder + "shap_feature_values_.csv", index=False)
        shap.summary_plot(shap_values, X_train.drop(cols_not_for_training, axis=1), plot_type="bar", show=False)
        plt.savefig(saving_folder + 'Shap_feature_importance.jpg', bbox_inches='tight', dpi=200)
        # shap.plots.bar(shap_values, show=False)
        # plt.savefig('Shap feature importance_2.jpg', bbox_inches='tight', dpi=200)
        print(f"Shap generation and saving took {time.time() -st} seconds")

    return results

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from core.classification import classify_by_XGBoost_v2  #, classify_by_FCN
from core.classification.evaluate_v2 import evaluate_clf_performance
from core.classification.learning_utils import encode_categorical_labels_to_numerical
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)


def classify(Xy, configs):

    global model, X_train
    Xy = Xy.dropna(subset=['Label'])
    labels = Xy['Label']
    y, labels_list, class_list = encode_categorical_labels_to_numerical(labels)
    # labels_list = np.unique(labels).tolist()

    X = Xy.drop('Label', axis=1)

    # ))))))))))))))))))))))))))))))))))))))))))))))
    # if configs['feature_extraction']['Concat_last_day']:
    #     X_prev = X.iloc[:-1, 7:-2].reset_index(drop=True)
    #     X_prev.columns = [f"prev_{x}" for x in X_prev.columns]
    #     X = pd.concat([X.iloc[1:, :].reset_index(drop=True), X_prev], axis=1)
    #     y = y[1:]
    # ))))))))))))))))))))))))))))))))))))))))))))))

    # Dummy columns (those that will not be used for training)
    cols_not_for_training = ["Ticker", 'Date', 'close', 'open', 'high', 'low', 'volume', 'Max_increase', 'Max_decrease']

    # style: method to divide data into train and test
    validation_style = configs['validation']['style']

    n_folds = 1
    if validation_style == 'hold_out':
        n_folds = 1  # number of repetition

    elif validation_style == "k_fold":
        n_folds = configs['validation']["n_folds"]
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds)
        skf_generator = skf.split(X, y)

    elif validation_style == "Group_Kfold":
        from sklearn.model_selection import GroupKFold
        n_folds = configs['validation']["n_folds"]
        group_col = configs['validation']['K_fold_based_on']
        groups = X[group_col]
        skf = GroupKFold(n_splits=n_folds)
        skf_generator = skf.split(X, y, groups)

    elif validation_style == "OOT" or validation_style == "Out-Of-Time":
        n_folds = 1

    # ------------------------------------------------------------------------------------------------------------
    # -------  split data into train and test and train and test a model in each time (each fold) ---------------
    train_ind = []
    test_ind = []
    indices = list(range(len(y)))

    result_folds_train = []
    result_folds_test = []

    for i in range(1, n_folds + 1):
        print(f"Fold {i} / {n_folds}")
        if validation_style == 'hold_out':
            if configs['validation'].get('shuffle', True):
                np.random.shuffle(indices)

            train_percent = configs['validation']['hold_out_train_percentage']
            if train_percent < 1:
                train_cut = round(len(indices) * train_percent)
                train_ind = (indices[:train_cut])
                test_ind = (indices[train_cut:])

            elif train_percent == 1:  # it's not reasonable to have all data for train, but just for test of few samples
                train_ind = (indices[:])
                test_ind = (indices[:])

        elif validation_style == "Kfold" or validation_style == "Group_Kfold":
            if configs['validation']['shuffle']:
                np.random.shuffle(indices)

            s = next(skf_generator)
            train_ind = s[0]
            test_ind = s[1]

        elif validation_style == "OOT" or validation_style == "Out-Of-Time":
            OOT_split_date = configs['validation']["OOT_split_date"]
            period = configs['Target_definition']['Period']
            OOT_split_date_test = datetime.strptime(OOT_split_date, "%Y-%m-%d") + timedelta(period+2)

            train_ind = np.where(X['Date'] < OOT_split_date)[0]
            test_ind = np.where(X['Date'] > OOT_split_date_test)[0]

        #
        X_train = X.iloc[train_ind, :]
        y_tr = y[train_ind]
        X_test = X.iloc[test_ind, :]
        y_ts = y[test_ind]

        #
        # --------------------------------------------  Scale features -------------------------------------------
        # X_train, X_test, scaler = scaling(X_train, X_test, configs["classification"]["features_scaling"])

        # ----------------------------- Classification of each fold begins here ----------------------------------
        classifier = configs["classification"]["classifier"]
        model = prob_y_tr = prob_y_ts = None

        #
        if classifier == "XGBoost":
            model, prob_y_tr, prob_y_ts = classify_by_XGBoost_v2.classify(
                X_train.drop(cols_not_for_training, axis=1), y_tr,
                X_test.drop(cols_not_for_training, axis=1), y_ts, configs)

        # if classifier == "FCN":
        #     model, prob_y_tr, prob_y_ts = classify_by_FCN.classify(
        #         X_train.drop(cols_not_for_training, axis=1), y_tr,
        #         X_test.drop(cols_not_for_training, axis=1), y_ts, configs)

        result_fold_tr = pd.DataFrame({'y_true': y_tr, 'y_pred': prob_y_tr, 'fold': np.ones(len(y_tr)) * i})
        # also add to above 'Max Increase': X_train['Max_increase'], 'Max_Decrease': X_train['Max_Decrease']})
        result_fold_ts = pd.DataFrame({'y_true': y_ts, 'y_pred': prob_y_ts, 'fold': np.ones(len(y_ts)) * i})
        result_fold_ts = pd.concat([X_test[cols_not_for_training].reset_index(drop=True),
                                    result_fold_ts.reset_index(drop=True)], axis=1)

        result_folds_train.append(result_fold_tr)
        result_folds_test.append(result_fold_ts)

        # --------------------------------- End of fold loop ----------------------------------

    print("Calculating Classification Performance Metrics ...")
    gr_truth_and_prediction_train = pd.concat(result_folds_train)
    gr_truth_and_prediction_test = pd.concat(result_folds_test)
    results_tr = evaluate_clf_performance(gr_truth_and_prediction_train,
                                          metrics=['AUPRC', 'Precision', 'Recall', 'Lift', 'AUC', 'Accuracy', 'CM'],
                                          top_n_percentiles=[0.5, 1, 3, 5, 10])
    results_ts = evaluate_clf_performance(gr_truth_and_prediction_test,
                                          metrics=['AUPRC', 'Precision', 'Recall', 'Lift', 'AUC', 'Accuracy', 'CM'],
                                          top_n_percentiles=[0.5, 1, 3, 5, 10])

    results = pd.DataFrame([results_tr, results_ts], index=['Train', 'Test'])
    if configs["save_the_results"]["save_json_file"]:
        results = save_outputs(results_tr, results_ts, gr_truth_and_prediction_train, gr_truth_and_prediction_test,
                               model, cols_not_for_training, configs)

    return results


def save_outputs(result_tr, result_ts, gr_truth_and_prediction_tr, gr_truth_and_prediction_ts, learner, cols_not_for_training, config):
    results = pd.DataFrame([result_tr, result_ts], index=['Train', 'Test'])

    # ------------------------------------- Save Results, Confusion Matrix and Model ---------------------------------
    saving_folder = config['saving_folder']
    results.to_csv(saving_folder + 'performances.csv')
    results_filename = saving_folder + "configs.json"
    with open(results_filename, 'w') as outfile:
        json.dump(config, outfile, separators=(',', ':'), indent=4)

    if config["save_the_results"]["save_post_analysis"] or config["save_the_results"]["save_detailed_results"]:
        gr_truth_and_prediction_ts.to_csv(saving_folder + 'test_detailed_result.csv', index=False)
        gr_truth_and_prediction_tr.to_csv(saving_folder + 'train_detailed_result.csv', index=False)

    if config["save_the_results"]["save_model"]:
        learner.save_model(saving_folder + "model_sklearn.json")  # save model in JSON format
        # model.save_model("model_sklearn.txt")                 # save model in text format

    if config["save_the_results"]["save_shap_feat_importance"]:
        import shap
        print('Generating shap values')
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

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
# from sklearn.metrics import roc_curve, recall_score, precision_score, auc


def evaluate_clf_performance(clf_outputs, metrics, top_n_percentiles=(1,)):
    """
    Function to calculate different performance metrics of a classifier:
        - clf_outputs: a dataframe with two columns: y_true and y_prediction
        - metrics: evaluation metrics, including AUROC, AUPRC, precision, recall, lift, etc.
        - top_n_percentiles: list of top n percentiles to be considered as positive labels (only usable
        for precision, recall, lift.
        -
    """
    y_true = clf_outputs['y_true']
    y_pred = clf_outputs['y_pred']

    results = {}

    if 'AUC' in metrics or 'AUROC' in metrics:
        results.update({"AUROC": round(roc_auc_score(y_true, y_pred), 3)})

    if 'AUPRC' in metrics:
        results.update({"AUPRC": round(average_precision_score(y_true, y_pred), 3)})

    if 'Precision' in metrics or 'Recall' in metrics or 'Lift' in metrics:
        for n in top_n_percentiles:
            precision, recall, lift, acc, cm = get_precision_recall_lift_accuracy(y_true, y_pred, n)
            if 'Precision' in metrics:
                results.update({f"Precision@{n}%": precision})
            if 'Recall' in metrics:
                results.update({f"Recall@{n}%": recall})
            if 'Lift' in metrics:
                results.update({f"Lift@{n}%": lift})
            if 'Accuracy' in metrics:
                results.update({f"Accuracy@{n}%": acc})
            if 'CM' in metrics:
                results.update({f"CM@{n}%": cm})

    return results  # pd.DataFrame([results], index=[''])


def get_prediction_labels(y_scores, N=1):
    """
    function to convert the model scores to 0/1 labels based on top N-percentile (i.e. top N
    percentile of scores will be considered as positive labels).

    Parameters:
        - N: number of top percentiles to be considered as positive labels.(optional, default=10)
    """
    top_n_perc_threshold = np.percentile(y_scores, 100 - N)
    pred_labels = y_scores.copy()
    pred_labels[pred_labels < top_n_perc_threshold] = 0
    pred_labels[pred_labels >= top_n_perc_threshold] = 1

    return pred_labels.astype(int)


def confusion_matrix_fast(y_true, y_pred):
    """
    Function to calculate the confusion matrix. This is faster and more efficient than the sklearn's function.
    Parameters:
        - y_true: Ground truth labels (list of zero's and one's)
        - y_pred: Predicted labels (list of zero's and one's)
    """
    y_pred = y_pred.astype(int)
    totaltrue = np.sum(y_true)
    total_false = len(y_true) - totaltrue
    total_positive = np.sum(y_pred)
    tp = np.sum(y_true & y_pred)
    fp = total_positive - tp

    return np.array([[total_false - fp, fp],  # true negatives, false positives
                     [totaltrue - tp, tp]])   # false negatives, true positives


def get_precision_recall_lift_accuracy(y_true, y_pred, N=1, get_labels=True):
    """
    Function to calculate the Precision, Recall, and lift at the top N percentile of the scores.
    Parameters:
        - N: number of top percentiles to be considered as positive labels.(optional, default=10).
        - get_labels: when True, the function will convert the model's output scores to binary 0/1 labels
          based on top N percentile.
          when False, it means that the model's output are already binary labels and therefore 'y_pred' will
          be used.
    """
    if get_labels:
        predicted_labels = get_prediction_labels(y_pred, N)
    else:
        predicted_labels = y_pred

    tn, fp, fn, tp = confusion_matrix_fast(y_true, predicted_labels).ravel()
    pos_fraction = y_true.mean()  # fraction of positive labels in the group
    precision = np.round(tp / (tp + fp), 3)  # positive predictive value (precision)
    recall = np.round(tp / (tp + fn), 3)  # true positive rate (recall)
    lift = np.round(precision / pos_fraction, 1)
    accuracy = np.round(100 * (tp + tn) / (tn + fp + fn + tp), 0)
    cm = [[tp, fp], [fn, tn]]

    return precision, recall, lift, accuracy, cm

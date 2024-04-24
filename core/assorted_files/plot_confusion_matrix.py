import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
# import matplotlib
# matplotlib.use('Agg')


def plotConfusionMatrix(title, trueY, predY, labels):
    testY_tmp = trueY.copy()
    testY_tmp.append('null')
    predY_tmp = predY.copy()
    predY_tmp.append('null')
    labels_tmp = labels.copy()
    labels_tmp.append('null')
    matrix = confusion_matrix(testY_tmp, predY_tmp)
    cm_normalized = np.divide(matrix, matrix.sum(axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_normalized)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + labels_tmp)
    ax.set_yticklabels([''] + labels_tmp)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.show()


def plot_confusion_matrix_v1(cm, classes, axes):
    # fig, ax = plt.subplots(dpi=110, figsize=(7.5, 7))
    axes[0].imshow(cm, interpolation='nearest', cmap="GnBu")

    tick_marks = np.arange(len(classes))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(classes, rotation=0, fontsize=16)
    axes[0].set_yticklabels(classes, fontsize=16)

    thresh = cm.max() / 2.
    n_samples = np.sum(cm, axis=1)
    cm_percent = np.round(100 * np.divide(cm, np.expand_dims(n_samples, 1)), 1).astype('str')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")
        axes[0].text(j, i+0.15, format("%" + cm_percent[i, j], 's'),
                     ha="center", va="center", fontsize=12,
                     color="white" if cm[i, j] > thresh else "black")

    acc = (cm.diagonal().sum())/np.sum(cm) * 100
    sens = (cm[0, 0])/(cm[0, 0] + cm[0, 1]) * 100
    prec = (cm[0, 0])/(cm[0, 0] + cm[1, 0]) * 100
    Fscore = 2 * (prec*sens) / (prec + sens)
    n_samples = np.sum(cm)
    fall_percentage = 100 * (cm[0, 0] + cm[0, 1]) / n_samples
    axes[0].set_title(f"#Samples={n_samples:3.0f}, Fall percentage: {fall_percentage:2.0f}% \n Acc= {acc:2.1f} , Sens.= {sens:2.1f}, Prec= {prec:2.1f}, Fscore={Fscore: 2.1f}", fontsize=14)

    # plt.tight_layout()
    return axes


def plot_confusion_matrix_v2(cm, classes, x_label='x_label', y_label='y_label', normalize=False,
                             title='Confusion matrix', cmap="GnBu"):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = cm

    # fig, ax = plt.subplots(dpi=110, figsize=(8, 7))  #
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center", fontsize=14,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.colorbar()

    plt.show()


def plot_confusion_matrix_v3(cm, classes, normalize=False, title='Confusion matrix', save_plot_conf_matrix=2, save_file_name='tmp'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = cm

    fig, ax = plt.subplots(dpi=110, figsize=(7.5, 7))
    ax.imshow(cm, interpolation='nearest', cmap="GnBu")
    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i-0.45, format(cm[i, j], fmt),
                 ha="center", va="top", fontsize=12,
                 color="white" if cm[i, j] > thresh else "black")

        # plt.text(j, i - 0.3, format(confidence_cm[i, j], '.2f'),
        #          ha="center", va="top", fontsize=10,
        #          color="white" if cm[i, j] > thresh else "black")

        # if i != j:
        #     plt.text(j, i-0.35, format(np.array2string(ids[i][j], separator=' , ', max_line_width=15)),
        #              ha="center", va="top", fontsize=6,
        #              color="k")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)
    if save_plot_conf_matrix > 0:

        # import time
        # acc = np.trace(cm) / np.sum(cm) * 100
        # filename = 'Results/%s/Acc= %d -%s.png' % (time.strftime("%Y.%m.%d"), acc*100, time.strftime("-%H.%M"))
        fig.savefig(save_file_name, bbox_inches='tight', dpi=300)

        if save_plot_conf_matrix == 2:
            plt.show()


def my_confusion_matrix(y_test, y_pred, n_classes, csi_id):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    cm = np.zeros((n_classes, n_classes))
    id_s = []
    indices = np.zeros((n_classes, n_classes, len(y_test)))
    for i in range(n_classes):
        row = []
        for j in range(n_classes):
            cm[i, j] = np.sum((y_test == i) * (y_pred == j))
            ind = (y_test == i) * (y_pred == j)
            indices[i, j, :] = ind
            row.append(csi_id[ind])
        id_s.append(row)

    # print(cm)
    return cm, indices, id_s


def my_confusion_matrix_v2(y_test, y_pred, classes, confidence):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    confidences_cm = np.zeros((n_classes, n_classes))
    id_s = []
    indices = np.zeros((n_classes, n_classes, len(y_test)))
    for i, c1 in enumerate(classes):
        row = []
        for j, c2 in enumerate(classes):
            a = (y_test == c1) * (y_pred == c2)
            cm[i, j] = np.sum((y_test == c1) * (y_pred == c2))
            confidences_cm[i, j] = max((0, np.mean(confidence[a])))
            ind = (y_test == c1) * (y_pred == c2)
            indices[i, j, :] = ind
        id_s.append(row)

    return cm, indices, id_s, confidences_cm


def plot_confusion_matrix_h(cm_raw, classes, x_label='Predicted label', y_label='True label', normalize=False,
                          title='Confusion matrix', cmap='Reds'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm_raw.sum(axis=0).astype('int')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = cm

    fig, axs = plt.subplots()
    axs.imshow(cm, cmap=cmap)
    axs.set_title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def save_confusion_matrix(cm, classes, save_file_name, x_label='x_label', y_label='y_label', y_ticks='', title="Confusion Matrix"):

    fig, axs = plt.subplots(dpi=110, figsize=(14, 7))
    axs.imshow(cm, interpolation='nearest', cmap="GnBu")

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center", fontsize=10,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.title(title)
    plt.ylabel(y_label, rotation=0)
    plt.xlabel(x_label)
    plt.yticks(np.arange(len(y_ticks)), y_ticks, fontsize=7)
    plt.savefig(save_file_name, bbox_inches='tight', dpi=300)

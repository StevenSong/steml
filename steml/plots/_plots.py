import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve


def plot_roc_curve(y_trues, y_preds, title, filename, names=None):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 101)
    for y_true, y_pred in zip(y_trues, y_preds):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1
    std_tpr = np.std(tprs, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.5)
    ax.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:0.3f} $\pm$ {std_auc:0.3f})')
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=f'$\pm$ 1 std. dev.')
    if names is not None:
        for name, tpr, _auc in zip(names, tprs, aucs):
            ax.plot(mean_fpr, tpr, label=f'{name} (AUC = {_auc:0.3f})')

    ax.legend(loc='lower right')
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_title(title)

    plt.savefig(filename)
    plt.show()


def plot_calibration_curve(y_trues, y_preds, title, filename):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5,10))

    briers = []
    for y_true, y_pred in zip(y_trues, y_preds):
        briers.append(brier_score_loss(y_true, y_pred))
    y_true = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
    mean_brier = np.mean(briers)
    std_brier = np.std(briers)

    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.5)
    ax1.plot(prob_pred, prob_true, '-o', label=f'Brier score {mean_brier:0.3f} $\pm$ {std_brier:0.3f}')
    ax1.legend(loc='lower right')
    ax1.set_xlim([-0.05,1.05])
    ax1.set_ylim([-0.05,1.05])
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("True Probability")
    ax1.set_title(title)

    ax2.hist(y_pred)
    ax2.set_xlim([-0.05,1.05])
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')

    plt.savefig(filename)
    plt.show()

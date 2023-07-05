import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report,
)


def generate_minimal_report(y_true, y_score, threshold=0.5):
    """ Evaluate a trained model using some data
    """
    y_pred = (y_score >= threshold).astype(int)
    report = '\n*** Using threshold = %s ***\n' % threshold
    report += classification_report(y_true, y_pred, zero_division=0)
    report += '\n*** AUROC-CI = %s ***\n' % auroc_ci(y_true, y_score)
    return report


def generate_report(y_prob_dev, y_prob_test, y_dev, y_test):
    """ Evaluate a trained model using the test data, after identifying the
        optimal threshold using the validation data
    """
    optimal_threshold = find_optimal_threshold(y_prob_dev, y_dev)
    report = ''
    for threshold in [0.5, optimal_threshold]:
        y_pred_test = (y_prob_test >= threshold).astype(int)
        report += '\n*** Using threshold = %s ***\n' % threshold
        report += classification_report(y_test, y_pred_test)
    report += '\n*** AUROC-CI = %s ***\n' % auroc_ci(y_test, y_prob_test)
    return report


def find_optimal_threshold(y_prob_dev, y_dev):
    """ Find optimal decision threshold using the validation set
    """
    thresholds = np.linspace(0, 1, 100)
    scores = []
    for t in thresholds:
        y_pred_dev = (y_prob_dev >= t).astype(int)
        score = f1_score(y_dev, y_pred_dev)
        scores.append(score)
    return thresholds[np.argmax(scores)]


def auroc_ci(y_true, y_score, t_value=1.96):
    """ Compute confidence interval of auroc score using Racha's method
    """
    auroc = roc_auc_score(y_true, y_score)
    n1 = sum(y_true == 1)
    n2 = sum(y_true == 0)
    p1 = (n1 - 1) * (auroc / (2 - auroc) - auroc ** 2)
    p2 = (n2 - 1) * (2 * auroc ** 2 / (1 + auroc) - auroc ** 2)
    std_auroc = ((auroc * (1 - auroc) + p1 + p2) / (n1 * n2)) ** 0.5
    low, high = (auroc - t_value * std_auroc, auroc + t_value * std_auroc)
    return '%.3f (%.3f, %.3f)' % (auroc, low, high)


def plot_roc_curve(y_true, y_scores):
    """ Plot ROC curve for a set of prediction scores, given true labels
    """
    # Compute ROC curve and AUROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig = plt.figure()
    label = 'ROC curve (area = %0.2f)' % roc_auc
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Return finalized figure
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc='lower right')
    return fig


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, weight=self.weight, reduction='none')
        exp_loss = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - exp_loss)**self.gamma * BCE_loss
        return focal_loss.mean()
    
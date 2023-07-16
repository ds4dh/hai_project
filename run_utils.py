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


def generate_dict_report(y_true, y_score, threshold=0.5):
    """ Evaluate a trained model using some data
    """
    # Base classification report (using 0.5 threshold)
    y_pred = (y_score >= 0.5).astype(int)
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True)
    report['threshold_base'] = 0.5
    
    # Classification report with threhsold optimized on f1-score
    y_pred_optim = (y_score >= threshold).astype(int)
    report_optim = classification_report(
        y_true, y_pred_optim, zero_division=0, output_dict=True)
    report.update({'%s_optim' % k: v for k, v in report_optim.items()})
    report['threshold_optim'] = threshold
    
    # Auroc and confidence interval
    auroc, auroc_low, auroc_high = auroc_ci(y_true, y_score)
    report['auroc'] = auroc
    report['auroc-low'] = auroc_low
    report['auroc-high'] = auroc_high
    
    # Return report
    return report


def find_optimal_threshold(y_true, y_score):
    """ Find optimal decision threshold (using the validation set)
    """
    thresholds = np.linspace(0, 1, 100)
    scores = []
    for t in thresholds:
        y_pred_dev = (y_score >= t).astype(int)
        score = f1_score(y_true, y_pred_dev)
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
    return (auroc, low, high)


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
    
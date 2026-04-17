import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title='Confusion Matrix'):
    """
    Plot a confusion matrix.
    Args:
        y_true: array of true labels (indices).
        y_pred: array of predicted labels (indices).
        class_names: list of class name strings for labeling the matrix axes.
        normalize: whether to normalize the confusion matrix by true row sums.
        title: title for the plot.
    Returns:
        fig, ax: Matplotlib figure and axis objects.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # Set tick labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    # Rotate x tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # Annotate cells with values
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

def plot_roc_curve(y_true, y_score, class_names, title='ROC Curve'):
    """
    Plot ROC curve(s) for binary or multi-class classification.
    Args:
        y_true: array of true labels (shape: n_samples,).
        y_score: array of predicted probabilities (shape: n_samples, n_classes).
        class_names: list of class names.
        title: title for the plot.
    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    if n_classes == 2:
        # Binary case: ROC for positive class (assumed index 1)
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1], pos_label=1,drop_intermediate=False)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
    else:
        # Multi-class: plot ROC for each class vs rest
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i], drop_intermediate=False)
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"Class {class_names[i]} (AUC = {roc_auc_val:.2f})")
        # Compute micro-average ROC (aggregate all classes)
        y_true_onehot = np.zeros((y_true.shape[0], n_classes))
        for i in range(n_classes):
            y_true_onehot[:, i] = (y_true == i).astype(int)
        fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, color='navy', lw=2, linestyle='--',
                label=f"Micro-average ROC (AUC = {roc_auc_micro:.2f})")
    # Plot chance line (diagonal)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax

def plot_pr_curve(y_true, y_score, class_names, title='Precision-Recall Curve'):
    """
    Plot Precision-Recall curve(s) for binary or multi-class classification.
    Args:
        y_true: array of true labels.
        y_score: array of predicted probabilities (shape: n_samples, n_classes).
        class_names: list of class names.
        title: title for the plot.
    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    if n_classes == 2:
        # Binary case: PR curve for positive class
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1], pos_label=1, drop_intermediate=False)
        ap_val = average_precision_score(y_true, y_score[:, 1], pos_label=1)
        ax.plot(recall, precision, color='darkorange', lw=2, label=f"PR curve (AP = {ap_val:.2f})")
        # Plot baseline (chance) as horizontal line at positive class frequency
        pos_rate = (y_true == 1).mean()
        ax.hlines(pos_rate, 0, 1, color='gray', linestyle='--', label=f"Chance (pos rate = {pos_rate:.2f})")
    else:
        # Multi-class: PR curve for each class
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score[:, i], drop_intermediate=False)
            ap_val = average_precision_score(y_true_binary, y_score[:, i])
            ax.plot(recall, precision, lw=2, label=f"Class {class_names[i]} (AP = {ap_val:.2f})")
        # Micro-average PR (all classes)
        y_true_onehot = np.zeros((y_true.shape[0], n_classes))
        for i in range(n_classes):
            y_true_onehot[:, i] = (y_true == i).astype(int)
        precision_micro, recall_micro, _ = precision_recall_curve(y_true_onehot.ravel(), y_score.ravel())
        ap_micro = average_precision_score(y_true_onehot, y_score, average='micro')
        ax.plot(recall_micro, precision_micro, color='navy', lw=2, linestyle='--',
                label=f"Micro-average PR (AP = {ap_micro:.2f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig, ax

def plot_attention_heatmap(attention_weights, title='Attention Weights'):
    """
    Plot a heatmap (as a color bar) of attention weights for a single sequence.
    Args:
        attention_weights: 1D array or list of attention weights (length = number of time steps).
        title: title for the plot.
    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    weights = np.array(attention_weights)
    fig, ax = plt.subplots(figsize=(max(6, weights.shape[0] // 2), 1.5))
    # Plot a single-row heatmap
    ax.imshow(weights[np.newaxis, :], aspect='auto', cmap='viridis')
    ax.set_xlabel('Time step')
    ax.set_yticks([])  # no y-axis labels (single row)
    ax.set_title(title)
    # Add a color bar to show scale
    fig.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.05)
    return fig, ax

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import torch
import numpy as np

def plot_loss_curve(epoch_losses, title="Training Loss Curve"):
    """
    Plots training loss over epochs.
    Args:
        epoch_losses (list or array): list of epoch losses
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8,4))
    plt.plot(epoch_losses, marker='o')
    # 🎯 UPDATED: Use the passed title argument
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_roc_curve(model, X_test, y_test, title='ROC Curve'):
    """
    Plots ROC curve and computes AUC.
    Args:
        model: trained PyTorch model
        X_test: test data as numpy array or torch tensor
        y_test: true labels as numpy array
        title (str): Title for the plot.
    """
    model.eval()
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        y_probs = model(X_test).flatten().numpy()
    
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, threshold=0.5, title="Confusion Matrix"):
    """
    Plots confusion matrix.
    Args:
        model: trained PyTorch model
        X_test: test data as numpy array or torch tensor
        y_test: true labels as numpy array
        threshold: probability threshold for classification
        title (str): Title for the plot.
    """
    model.eval()
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        y_probs = model(X_test).flatten().numpy()
    
    y_pred = (y_probs > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    # 🎯 UPDATED: Use the passed title argument
    plt.title(title)
    plt.show()
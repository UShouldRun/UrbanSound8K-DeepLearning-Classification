import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import gc
import matplotlib.pyplot as plt
import torch

def read_config(config_path: str) -> dict:
    """
    Reads a JSON file and returns the data as a dictionary.
    """
    with open(config_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def load_fold_paths(data_cache_dir):
    """
    Returns:
      train_paths[fold] = [list of paths]
      train_labels[fold] = [list of labels]
    """
    train_paths = {}
    train_labels = {}

    test_paths = {}
    test_labels = {}

    for f in range(1, 11):

        train_dir = os.path.join(data_cache_dir, f"fold_{f}", "train")
        with open(os.path.join(train_dir, "labels.json"), "r") as fjson:
            labels = json.load(fjson)

        paths = [
            os.path.join(train_dir, fname)
            for fname in sorted(os.listdir(train_dir))
            if fname.endswith(".npy")
        ]

        train_paths[f] = paths
        train_labels[f] = labels

        test_dir = os.path.join(data_cache_dir, f"fold_{f}", "test")
        with open(os.path.join(test_dir, "labels.json"), "r") as fjson:
            labels = json.load(fjson)

        paths = [
            os.path.join(test_dir, fname)
            for fname in sorted(os.listdir(test_dir))
            if fname.endswith(".npy")
        ]

        test_paths[f] = paths
        test_labels[f] = labels

    return train_paths, train_labels, test_paths, test_labels

def compute_metrics(predictions, targets, num_classes=10):
    """
    Compute comprehensive metrics
    
    Parameters:
    -----------
    predictions : array-like
        Predicted labels
    targets : array-like
        True labels
    num_classes : int
        Number of classes
    
    Returns:
    --------
    dict : Dictionary with all metrics
    """
    # Accuracy
    accuracy = (np.array(predictions) == np.array(targets)).mean()
    
    # Precision, Recall, F1 (macro and weighted)
    precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
    
    recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
    
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(targets, predictions, average=None, zero_division=0, labels=range(num_classes))
    recall_per_class = recall_score(targets, predictions, average=None, zero_division=0, labels=range(num_classes))
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0, labels=range(num_classes))
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }

def cleanup(train_loader, val_loader, test_loader,
            train_dataset, val_dataset, test_dataset,
            model, optimizer, criterion, history,
            all_predictions, all_targets,
            checkpoint):

    del train_loader, val_loader, test_loader
    gc.collect()  # Force garbage collection immediately after deleting loaders
    
    del train_dataset, val_dataset, test_dataset
    
    del model, optimizer, criterion, history
    del all_predictions, all_targets
    del checkpoint  
    
    gc.collect()
    gc.collect()  # Call twice for good measure
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for GPU operations to complete

    plt.close('all')
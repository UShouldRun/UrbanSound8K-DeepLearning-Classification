import os
import gc

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import json

from sklearn.metrics import precision_score, recall_score, f1_score

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
    
def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way"""
    print(f"\n{prefix}Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Precision (wtd):   {metrics['precision_weighted']:.4f}")
    print(f"  Recall (wtd):      {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (wtd):    {metrics['f1_weighted']:.4f}")

def print_cross_validation_results(fold_results, save_dir, cumulative_confusion_matrix, all_fold_accuracies, config):
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    mean_accuracy = np.mean([r['test_accuracy'] for r in fold_results])
    std_accuracy = np.std([r['test_accuracy'] for r in fold_results])
    
    mean_precision_macro = np.mean([r['test_precision_macro'] for r in fold_results])
    std_precision_macro = np.std([r['test_precision_macro'] for r in fold_results])
    
    mean_recall_macro = np.mean([r['test_recall_macro'] for r in fold_results])
    std_recall_macro = np.std([r['test_recall_macro'] for r in fold_results])
    
    mean_f1_macro = np.mean([r['test_f1_macro'] for r in fold_results])
    std_f1_macro = np.std([r['test_f1_macro'] for r in fold_results])
    
    mean_precision_weighted = np.mean([r['test_precision_weighted'] for r in fold_results])
    std_precision_weighted = np.std([r['test_precision_weighted'] for r in fold_results])
    
    mean_recall_weighted = np.mean([r['test_recall_weighted'] for r in fold_results])
    std_recall_weighted = np.std([r['test_recall_weighted'] for r in fold_results])
    
    mean_f1_weighted = np.mean([r['test_f1_weighted'] for r in fold_results])
    std_f1_weighted = np.std([r['test_f1_weighted'] for r in fold_results])
    
    print(f"\nMetrics per fold:")
    print(f"{'Fold':<6} {'Acc':<8} {'Prec(M)':<10} {'Rec(M)':<10} {'F1(M)':<10}")
    print("-" * 50)
    for i, result in enumerate(fold_results, 1):
        print(f"{i:<6} {result['test_accuracy']:<8.4f} "
              f"{result['test_precision_macro']:<10.4f} "
              f"{result['test_recall_macro']:<10.4f} "
              f"{result['test_f1_macro']:<10.4f}")
    
    print("\n" + "="*60)
    print("AGGREGATE RESULTS (Mean ± Std)")
    print("="*60)
    print(f"Accuracy:              {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision (macro):     {mean_precision_macro:.4f} ± {std_precision_macro:.4f}")
    print(f"Recall (macro):        {mean_recall_macro:.4f} ± {std_recall_macro:.4f}")
    print(f"F1-Score (macro):      {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")
    print(f"Precision (weighted):  {mean_precision_weighted:.4f} ± {std_precision_weighted:.4f}")
    print(f"Recall (weighted):     {mean_recall_weighted:.4f} ± {std_recall_weighted:.4f}")
    print(f"F1-Score (weighted):   {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
    print("="*60)
    
    plot_confusion_matrix(
        cumulative_confusion_matrix,
        save_path=os.path.join(save_dir, 'cumulative_confusion_matrix.png'),
        title='Cumulative Confusion Matrix (10 Folds)',
        normalize=False
    )
    
    plot_confusion_matrix(
        cumulative_confusion_matrix,
        save_path=os.path.join(save_dir, 'cumulative_confusion_matrix_normalized.png'),
        title='Cumulative Confusion Matrix (Normalized)',
        normalize=True
    )
    
    final_results = {
        'fold_results': fold_results,
        'mean_accuracy': float(mean_accuracy),
        'std_accuracy': float(std_accuracy),
        'mean_precision_macro': float(mean_precision_macro),
        'std_precision_macro': float(std_precision_macro),
        'mean_recall_macro': float(mean_recall_macro),
        'std_recall_macro': float(std_recall_macro),
        'mean_f1_macro': float(mean_f1_macro),
        'std_f1_macro': float(std_f1_macro),
        'mean_precision_weighted': float(mean_precision_weighted),
        'std_precision_weighted': float(std_precision_weighted),
        'mean_recall_weighted': float(mean_recall_weighted),
        'std_recall_weighted': float(std_recall_weighted),
        'mean_f1_weighted': float(mean_f1_weighted),
        'std_f1_weighted': float(std_f1_weighted),
        'all_fold_accuracies': [float(a) for a in all_fold_accuracies],
        'cumulative_confusion_matrix': cumulative_confusion_matrix.tolist(),
        'config': config
    }
    
    with open(os.path.join(save_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    with open(os.path.join(save_dir, 'summary_report.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("10-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("Aggregate Results (Mean ± Std):\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:              {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Precision (macro):     {mean_precision_macro:.4f} ± {std_precision_macro:.4f}\n")
        f.write(f"Recall (macro):        {mean_recall_macro:.4f} ± {std_recall_macro:.4f}\n")
        f.write(f"F1-Score (macro):      {mean_f1_macro:.4f} ± {std_f1_macro:.4f}\n")
        f.write(f"Precision (weighted):  {mean_precision_weighted:.4f} ± {std_precision_weighted:.4f}\n")
        f.write(f"Recall (weighted):     {mean_recall_weighted:.4f} ± {std_recall_weighted:.4f}\n")
        f.write(f"F1-Score (weighted):   {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}\n")
        f.write("\n" + "="*60 + "\n\n")
        
        f.write("Per-Fold Results:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Fold':<6} {'Acc':<8} {'Prec(M)':<10} {'Rec(M)':<10} {'F1(M)':<10}\n")
        f.write("-"*60 + "\n")
        for i, result in enumerate(fold_results, 1):
            f.write(f"{i:<6} {result['test_accuracy']:<8.4f} "
                   f"{result['test_precision_macro']:<10.4f} "
                   f"{result['test_recall_macro']:<10.4f} "
                   f"{result['test_f1_macro']:<10.4f}\n")
    
    plot_fold_accuracies(
        all_fold_accuracies,
        save_path=os.path.join(save_dir, 'fold_accuracies.png')
    )
    
    plot_all_metrics_across_folds(
        fold_results,
        save_path=os.path.join(save_dir, 'all_metrics_across_folds.png')
    )
    
    print(f"\nAll results saved to: {save_dir}")
    
    return {
        'fold_results': fold_results,
        'cumulative_confusion_matrix': cumulative_confusion_matrix,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_precision_macro': mean_precision_macro,
        'std_precision_macro': std_precision_macro,
        'mean_recall_macro': mean_recall_macro,
        'std_recall_macro': std_recall_macro,
        'mean_f1_macro': mean_f1_macro,
        'std_f1_macro': std_f1_macro,
        'save_dir': save_dir
    }

def plot_confusion_matrix(cm, save_path=None, title='Confusion Matrix', normalize=False):
    """Plot confusion matrix"""
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_fold_accuracies(accuracies, save_path=None):
    """Plot accuracy across folds"""
    
    plt.figure(figsize=(10, 6))
    folds = list(range(1, len(accuracies) + 1))
    
    plt.plot(folds, accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.4f}')
    
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Test Accuracy Across 10 Folds', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(folds)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    
def plot_all_metrics_across_folds(fold_results, save_path=None):
    """Plot accuracy, precision, recall, and F1-score across folds."""

    # Extract metrics
    accuracies = [fr['test_accuracy'] for fr in fold_results]
    precision_macro = [fr['test_precision_macro'] for fr in fold_results]
    recall_macro = [fr['test_recall_macro'] for fr in fold_results]
    f1_macro = [fr['test_f1_macro'] for fr in fold_results]

    folds = list(range(1, len(fold_results) + 1))

    plt.figure(figsize=(12, 8))

    # Plot each metric
    plt.plot(folds, accuracies, marker='o', linewidth=2, label='Accuracy')
    plt.plot(folds, precision_macro, marker='o', linewidth=2, label='Precision (Macro)')
    plt.plot(folds, recall_macro, marker='o', linewidth=2, label='Recall (Macro)')
    plt.plot(folds, f1_macro, marker='o', linewidth=2, label='F1 Score (Macro)')

    # Add mean horizontal lines
    plt.axhline(np.mean(accuracies), linestyle='--', linewidth=1,
                label=f"Mean Accuracy: {np.mean(accuracies):.4f}")
    plt.axhline(np.mean(precision_macro), linestyle='--', linewidth=1,
                label=f"Mean Precision(M): {np.mean(precision_macro):.4f}")
    plt.axhline(np.mean(recall_macro), linestyle='--', linewidth=1,
                label=f"Mean Recall(M): {np.mean(recall_macro):.4f}")
    plt.axhline(np.mean(f1_macro), linestyle='--', linewidth=1,
                label=f"Mean F1(M): {np.mean(f1_macro):.4f}")

    # Labels and formatting
    plt.xlabel("Fold", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title("Performance Metrics Across Folds", fontsize=14, fontweight='bold')
    plt.xticks(folds)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save and show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
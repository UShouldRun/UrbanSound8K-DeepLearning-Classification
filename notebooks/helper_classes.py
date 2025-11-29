import torch
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class TrainingHistory:
    def __init__(self):
        self.history = defaultdict(list)
    
    def add(self, key, value):
        self.history[key].append(value)
    
    def plot(self, save_path=None):
        metrics = list(self.history.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            values = self.history[metric]
            axes[idx].plot(values, marker='o', linewidth=2)
            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch', fontsize=10)
            axes[idx].set_ylabel('Value', fontsize=10)
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def get_best(self, metric, mode='max'):
        values = self.history[metric]
        if mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value) + 1
        else:
            best_value = min(values)
            best_epoch = values.index(best_value) + 1
        return best_value, best_epoch
    
    def save(self, path):
        """Save history to file"""
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

class AudioDataset(Dataset):
    """
    Dataset class for audio data
    """
    def __init__(self, audio_data, labels, transform=None):
        """
        Parameters:
        -----------
        audio_data : list or np.ndarray
            List of audio arrays
        labels : list or np.ndarray
            Corresponding labels
        transform : callable, optional
            Optional transform to apply
        """
        self.audio_data = audio_data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        label = self.labels[idx]
        
        # Convert to tensor if not already
        if not isinstance(audio, torch.Tensor):
            audio = torch.FloatTensor(audio)
        
        if not isinstance(label, torch.Tensor):
            label = torch.LongTensor([label])[0]
        
        if self.transform:
            audio = self.transform(audio)
        
        return audio, label
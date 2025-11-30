import torch
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Spectrogram, MelSpectrogram, TimeStretch, AmplitudeToDB
from torch.distributions import Uniform

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
    
def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length

class RandomTimeStretch(TimeStretch):
    def __init__(self, max_perc, hop_length=None, n_freq=201, fixed_rate=None):
        super(RandomTimeStretch, self).__init__(hop_length, n_freq, fixed_rate)
        self._dist = Uniform(1.-max_perc, 1+max_perc)

    def forward(self, x):
        rate = self._dist.sample().item()
        return super(RandomTimeStretch, self).forward(x, rate), rate

class SpecNormalization(nn.Module):
    def __init__(self, norm_type, top_db=80.0):
        super(SpecNormalization, self).__init__()

        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x
        
    def z_transform(self, x):
        # Independent mean, std per batch
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean)/std 
        return x

    def forward(self, x):
        return self._norm(x)

class MelspectrogramStretch(MelSpectrogram):
    def __init__(self, hop_length=None, 
                       sample_rate=44100, 
                       num_mels=128, 
                       fft_length=2048, 
                       norm='whiten', 
                       stretch_param=[0.4, 0.4]):

        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate, 
                                                    n_fft=fft_length, 
                                                    hop_length=hop_length, 
                                                    n_mels=num_mels)

        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length, pad=self.pad, 
                                       power=None, normalized=False)

        # Augmentation
        self.prob = stretch_param[0]
        self.random_stretch = RandomTimeStretch(stretch_param[1], 
                                                self.hop_length, 
                                                self.n_fft//2+1, 
                                                fixed_rate=None)
        
        # Normalization (post spec processing)
        self.complex_norm = lambda x: torch.abs(x).pow(2.0)
        self.norm = SpecNormalization(norm)

    def forward(self, x, lengths=None):
        x = self.stft(x)

        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft//2)
            lengths = lengths.long()
        
        if torch.rand(1)[0] <= self.prob and self.training:
            # Stretch spectrogram in time using Phase Vocoder
            x, rate = self.random_stretch(x)
            # Modify the rate accordingly
            lengths = (lengths.float()/rate).long()+1
        
        x = self.complex_norm(x)
        x = self.mel_scale(x)

        # Normalize melspectrogram
        x = self.norm(x)

        if lengths is not None:
            return x, lengths        
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

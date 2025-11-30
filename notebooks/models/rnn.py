import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from tqdm import tqdm
from torch.utils.data import Dataset
from .helper_classes import MelspectrogramStretch

class AudioRNN(nn.Module):
    def __init__(self, num_classes=10, config=None):
        super(AudioRNN, self).__init__()
        
        self.num_classes = num_classes
        self.config = config if config else {}
        
        # Spectrogram parameters
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.num_mels = self.config.get('num_mels', 128)
        self.fft_length = self.config.get('fft_length', 2048)
        self.hop_length = self.config.get('hop_length', 512)
        
        # Melspectrogram transform
        self.spec = MelspectrogramStretch(
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            num_mels=self.num_mels,
            fft_length=self.fft_length,
            norm=self.config.get('norm', 'whiten'),
            stretch_param=self.config.get('stretch_param', [0.4, 0.4])
        )
        
        # RNN parameters
        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.bidirectional = self.config.get('bidirectional', False)
        self.rnn_dropout = self.config.get('rnn_dropout', 0.1)
        self.scheduler_step_size = self.config.get('scheduler_step_size', 5)
        self.scheduler_gamma = self.config.get('scheduler_gamma', 0.5)
        
        # Build LSTM
        self.lstm = nn.LSTM(
            input_size=self.num_mels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.rnn_dropout if self.num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Dense layer parameters
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.dropout_p = self.config.get('dropout', 0.3)
        
        # Build dense layers
        self.dense = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.LayerNorm(lstm_output_size),
            nn.Linear(lstm_output_size, num_classes)
        )
    
    def _many_to_one(self, t, lengths):
        """Extract last relevant output for each sequence"""
        return t[torch.arange(t.size(0)), lengths - 1]
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time) audio waveform
            lengths: (batch,) sequence lengths
        """
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        
        # Spectrogram: (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)
        
        # Transpose to (batch, time, channel, freq)
        x = x.transpose(1, -1)
        batch_size, time = x.size()[:2]
        
        # Flatten: (batch, time, features)
        x = x.reshape(batch_size, time, -1)
        
        # Pack padded sequence
        x_pack = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        x_pack, _ = self.lstm(x_pack)
        
        # Unpack
        x, _ = nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # Many-to-one
        x = self._many_to_one(x, lengths)
        
        # Dense layers
        x = self.dense(x)
        
        return F.log_softmax(x, dim=1)
    

    @staticmethod
    def trainModel(model, train_loader, val_loader, test_fold,
              history, config, fold_dir,
              epochs,
              optimizer, criterion, device):

        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=model.scheduler_step_size, gamma=model.scheduler_gamma)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
            
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc='Training', leave=False)
            for batch_idx, (audio, lengths, labels) in enumerate(train_pbar):
                audio = audio.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(audio, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = outputs.argmax(dim=1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })
                
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc='Validation', leave=False)
                for audio, lengths, labels in val_pbar:
                    audio = audio.to(device)
                    lengths = lengths.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(audio, lengths)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    pred = outputs.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            history.add('train_loss', avg_train_loss)
            history.add('train_accuracy', train_acc)
            history.add('val_loss', avg_val_loss)
            history.add('val_accuracy', val_acc)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                torch.save({
                    'fold': test_fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_acc': best_val_acc,
                    'config': config
                }, os.path.join(fold_dir, 'best_model.pth'))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step shceduler at the end of epoch
            scheduler.step()

        return best_val_loss, best_val_acc
    
    @staticmethod
    def test(model, criterion, test_loader, device):
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Testing')
            for audio, lengths, labels in test_pbar:
                audio = audio.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                outputs = model(audio, lengths)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                pred = outputs.argmax(dim=1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        
        return avg_test_loss, test_acc, all_predictions, all_targets

class LazyAudioRNNDataset(Dataset):
    def __init__(self, audio_paths, labels):
        """
        audio_paths : list of file paths to .npy audio files
        labels      : list/array of labels (string or int)
        """
        self.audio_paths = audio_paths

        labels = np.array(labels)

        if labels.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(labels)
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self.labels = np.array([self.label_to_idx[l] for l in labels], dtype=np.int64)
            print(f"Converted string labels to integers: {self.label_to_idx}")

        else:
            self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # ---- Lazy load from .npy instead of loading everything in RAM ----
        audio = np.load(self.audio_paths[idx], mmap_mode="r")

        label = self.labels[idx]

        if not isinstance(audio, torch.Tensor):
            audio = torch.FloatTensor(audio)

        if audio.dim() > 1:
            audio = audio.squeeze()

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return audio, label
    
def collate_fn_rnn(batch):
    """
    Input: List of (audio, label) tuples from dataset
    Output: Padded batch ready for RNN
    """
    audios, labels = zip(*batch)  # Separate audio and labels
    
    # 1. Get actual lengths of each audio
    lengths = torch.LongTensor([len(a) for a in audios])
    # Example: [44100, 88200, 22050]
    
    # 2. Pad all audios to the same length (longest in batch)
    padded_audio = nn.utils.rnn.pad_sequence(audios, batch_first=True)
    # Example shape: (3, 88200) - all padded to longest (88200)
    
    # 3. Stack labels
    labels = torch.stack(labels)
    
    return padded_audio, lengths, labels

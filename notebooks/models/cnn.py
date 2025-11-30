from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from .helper_classes import MelspectrogramStretch

class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, config={}):
        super(AudioCNN, self).__init__()
        
        self.num_classes = num_classes
        self.config = config

        self.net = nn.ModuleDict() 

        self.input_channels = 1

        # Spectrogram transform
        self.spec = MelspectrogramStretch(
            hop_length=config.get('hop_length', None),
            num_mels=config.get('num_mels', 128),
            fft_length=config.get('fft_length', 2048),
            norm=config.get('norm', 'whiten'),
            stretch_param=config.get('stretch_param', [0.4, 0.4])
        )

        # CNN parameters
        self.hidden_channels = self.config.get('hidden_channels', 32)
        self.num_layers = self.config.get('num_layers', 3)
        self.cnn_dropout = self.config.get('cnn_dropout', 0.3)

        self.dropout = self.config.get('dropout', 0.3)

        self.scheduler_step_size = self.config.get('scheduler_step_size', 5)
        self.scheduler_gamma = self.config.get('scheduler_gamma', 0.5)

        # self.final_flatten_size = self.config.get('final_flatten_size')
        self.padding = self.config.get('padding', 1)

        # Build network from cfg
        # Input shape: [channel, frequency, time]
        self.net['convs'] = nn.Sequential(
            # Layer 1: Input channel 1 -> Hidden channels
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, padding=self.padding),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=self.dropout),

            # Layer 2 to n_layers+1: All take hidden_channels -> Hidden channels
            *[nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=self.padding),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=self.dropout)
            ) for _ in range(self.num_layers)],
            
            # Final Pooling
            # nn.AvgPool2d(kernel_size=(2, 2)),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)) 
        )

        # Calculate the flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.spec.n_mels, 400)
            dummy_output = self.net['convs'](dummy_input)
            self.final_flatten_size = dummy_output.view(1, -1).size(1)
        
        # --- 3. Classification Head ---
        # The first Linear layer size must match the output size of the CNN body after flattening.
        # final_flatten_size needs to be calculated dynamically or estimated. 
        # For simplicity, we use a placeholder and a dense head.
        self.net['dense'] = nn.Sequential(
            nn.Linear(self.final_flatten_size, self.hidden_channels), 
            nn.ReLU(),
            nn.Dropout(p=self.cnn_dropout),
            nn.Linear(self.hidden_channels, self.num_classes) # Final output layer for classification
        )

    def forward(self, audio, lengths):
        # Add channel dimension: (batch, time) -> (batch, 1, time)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio = audio.float()

        # Compute mel spectrogram: (batch, 1, time) -> (batch, 1, freq, time)
        x, lengths = self.spec(audio, lengths)

        # CNN processing: (batch, channel, freq, time)
        x = self.net['convs'](x)

        # Flatten: (batch, time*freq*channel)
        x = x.view(x.size(0), -1)

        # Classification: (batch, classes)
        x = self.net['dense'](x)

        # Return raw logits for Cross Entropy Loss
        return x
    
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

            # TRAIN
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc='Training', leave=False)
            for audio, lengths, labels in train_pbar:

                audio = audio.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(audio, lengths)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                pred = outputs.argmax(dim=1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # VALIDATE
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
            
            # Log metrics
            history.add('train_loss', avg_train_loss)
            history.add('train_accuracy', train_acc)
            history.add('val_loss', avg_val_loss)
            history.add('val_accuracy', val_acc)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model for this fold
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
                print(f"✓ Best model saved (Val Loss: {best_val_loss:.4f})")

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
    
    @staticmethod
    def loadModel(path: str, config, device=None) -> nn.Module:
        if device is None:
            device = AudioCNN.setDevice()

        model = AudioCNN(num_classes = config["num_classes"], config=config)

        model.load_state_dict(torch.load(path, weights_only=True), strict = False)
        model.eval()
        print(f"Model loaded from {path}")
        print(model)
    
        return model
    
    # Set the device for training (GPU if available, else CPU)
    @staticmethod
    def setDevice() -> torch.device:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if torch.cuda.is_available():
            print(f"Using {device} device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"Using {device} device")
        return device

class LazyAudioCNNDataset(Dataset):
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
    
def collate_fn_cnn(batch):
    """
    Input: List of (audio, label) tuples from dataset
    Output: Padded batch ready for CNN
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

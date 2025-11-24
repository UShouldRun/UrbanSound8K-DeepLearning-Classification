import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

from .audio import MelspectrogramStretch
from torchparse import parse_cfg

class AudioCNN(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(AudioCNN, self).__init__(config)
        
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1

        self.classes = classes

        # Spectrogram transform
        self.spec = MelspectrogramStretch(
            hop_length=config.get('hop_length', None),
            num_mels=config.get('num_mels', 128),
            fft_length=config.get('fft_length', 2048),
            norm=config.get('norm', 'whiten'),
            stretch_param=config.get('stretch_param', [0.4, 0.4])
        )

        # Build network from cfg
        features = self.spec.n_mels  # features per time step
        self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.n_mels, 400])
        # self.net = parse_cfg(config['cfg'], in_shape=[400, features])

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, batch):
        x, lengths, _ = batch
        # x -> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)

        # (batch, channel, freq, time)
        x = self.net['convs'](x)

        # x -> (batch, time*freq*channel)
        x = x.view(x.size(0), -1)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, batch):
        with torch.no_grad():
            out_raw = self.forward(batch)
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()
            return self.classes[max_ind], out[:, max_ind].item()

class AudioRNN(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(AudioRNN, self).__init__(config)
        
        self.classes = classes

        # Spectrogram transform
        self.spec = MelspectrogramStretch(
            hop_length=config.get('hop_length', None),
            num_mels=config.get('num_mels', 128),
            fft_length=config.get('fft_length', 2048),
            norm=config.get('norm', 'whiten'),
            stretch_param=config.get('stretch_param', [0.4, 0.4])
        )
        
        # Build network from cfg
        # Only RNN + dense, no CNNs
        features = self.spec.n_mels  # features per time step
        self.net = parse_cfg(config['cfg'], in_shape=[400, features])

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def forward(self, batch):
        x, lengths, _ = batch  # unpack sequences, lengths, srs

        # (batch, channel, time)
        x = x.float().transpose(1, 2)

        # spectrogram -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)

        # flatten channel and freq for RNN input -> (batch, time, features)
        x = x.transpose(1, -1)
        batch_size, time = x.size()[:2]
        x = x.reshape(batch_size, time, -1)

        # pack padded sequence
        x_pack = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # pass through RNN (vanilla)
        x_pack, _ = self.net['recur'](x_pack)

        # unpack sequence
        x, _ = nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)

        # many-to-one (take last relevant output for each sequence)
        x = self._many_to_one(x, lengths)

        # dense layer for classification
        x = self.net['dense'](x)

        return F.log_softmax(x, dim=1)

    def predict(self, batch):
        with torch.no_grad():
            out_raw = self.forward(batch)
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()
            return self.classes[max_ind], out[:, max_ind].item()

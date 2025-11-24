import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from .base_model import BaseModel
from .audio import MelspectrogramStretch
from torchparse import parse_cfg

class RNN(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(RNN, self).__init__(config)
        
        # Determine input channels
        in_chan = 2 if config.get('transforms', {}).get('args', {}).get('channels', 'mono') == 'stereo' else 1
        
        self.classes = classes
        self.lstm_units = config.get('lstm_units', 64)
        self.lstm_layers = config.get('lstm_layers', 2)

        # Spectrogram transform
        self.spec = MelspectrogramStretch(
            hop_length=config.get('hop_length', None),
            num_mels=config.get('num_mels', 128),
            fft_length=config.get('fft_length', 2048),
            norm=config.get('norm', 'whiten'),
            stretch_param=config.get('stretch_param', [0.4, 0.4])
        )
        
        # Build network from cfg
        self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.n_mels, 400])
        
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]

        for _, layer in self.net['convs'].named_children():
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size, layer.stride])
                lengths = ((lengths + 2*p - k)//s + 1).long()
        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):
        x, lengths, _ = batch  # unpack sequences, lengths, srs

        # (batch, channel, time)
        x = x.float().transpose(1, 2)
        # spectrogram -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)
        # CNN layers
        x = self.net['convs'](x)
        lengths = self.modify_lengths(lengths)
        # (batch, time, freq*channel)
        x = x.transpose(1, -1)
        batch_size, time = x.size()[:2]
        x = x.reshape(batch_size, time, -1)
        x_pack = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        # LSTM
        x_pack, _ = self.net['recur'](x_pack)
        x, _ = nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        # many-to-one
        x = self._many_to_one(x, lengths)
        # Dense
        x = self.net['dense'](x)
        return F.log_softmax(x, dim=1)

    def predict(self, batch):
        with torch.no_grad():
            out_raw = self.forward(batch)
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()
            return self.classes[max_ind], out[:, max_ind].item()

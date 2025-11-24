import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

# F.max_pool2d needs kernel_size and stride. If only one argument is passed, 
# then kernel_size = stride

from .audio import MelspectrogramStretch
from torchparse import parse_cfg

# Architecture inspiration from: https://github.com/keunwoochoi/music-auto_tagging-keras
class AudioCRNN(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(AudioCRNN, self).__init__(config)
        
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1

        self.classes = classes
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, 
                                num_mels=128, 
                                fft_length=2048, 
                                norm='whiten', 
                                stretch_param=[0.4, 0.4])

        # shape -> (channel, freq, token_time)
        self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.n_mels, 400])

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.net['convs'].named_children():
            #if name.startswith(('conv2d','maxpool2d')):
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = ((lengths + 2*p - k)//s + 1).long()

        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs
        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)
        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)                

        # (batch, channel, freq, time)
        xt = self.net['convs'](xt)
        lengths = self.modify_lengths(lengths)

        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        with torch.no_grad():
            out_raw = self.forward( x )
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()        
            return self.classes[max_ind], out[:,max_ind].item()


class AudioCNN(AudioCRNN):

    def forward(self, batch):
        x, _, _ = batch
        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x = self.spec(x)                

        # (batch, channel, freq, time)
        x = self.net['convs'](x)

        # x -> (batch, time*freq*channel)
        x = x.view(x.size(0), -1)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x

class AudioRNN(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(AudioRNN, self).__init__(config)
        
        self.classes = classes
        self.rnn_units = config.get('rnn_units', 64)
        self.rnn_layers = config.get('rnn_layers', 2)
        self.bidirectional = config.get('bidirectional', True)

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

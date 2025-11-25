import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download()
dataset.validate()

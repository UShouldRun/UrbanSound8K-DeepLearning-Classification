import os

import torch
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, linear_relu_stack, name: str = "nn_model"):
        super().__init__()
        self.flatten = nn.Flatten()  # keeps batch dimension

        self.linear_relu_stack = linear_relu_stack

        self.name = name

    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)                

        # x -> (batch, time, freq, channel)
        x = x.transpose(1, -1)

        # x -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x

    # Instantiate a NN on the specified device
    @staticmethod
    def getModel(input_size: int, output_size: int, n_layers: int, start_units: int, divide_rate: float, device: torch.device) -> nn.Module:
        layers = NeuralNetwork.generateLayers(input_size, output_size, n_layers, start_units, divide_rate)
        model = NeuralNetwork(layers).to(device)
        print(model)
        return model

    # Crate Layers for the NN
    @staticmethod
    def generateLayers(input_size: int, output_size: int, n_layers: int, start_units: int = 128, divide_rate: float = 2.0):
        layers = []
        in_features = input_size
        out_features = start_units

        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        for i in range(n_layers - 1):
            next_features = int(out_features / divide_rate) 
            layers.append(nn.Linear(out_features, next_features))
            layers.append(nn.ReLU())
            out_features = next_features

        layers.append(nn.Linear(out_features, output_size))

        return nn.Sequential(*layers)

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

    # Binary Cross Entropy With Logits Loss
    # "This loss combines a Sigmoid layer and the BCELoss in one single class"
    # "This version is more numerically stable than using a plain Sigmoid followed by a BCELoss"
    @staticmethod
    def getDefaultCriterion():
        return nn.BCEWithLogitsLoss()

    # Adam Optimizer
    # TODO: SDG - stocastic gradial descent
    @staticmethod
    def getDefaultOptimizer(model: nn.Module, lr: float = 1e-3):
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        return optim.Adam(model.parameters(), lr=lr)

    @staticmethod
    def trainModel(model: nn.Module, train_loader: DataLoader, optimizer, criterion, epochs: int = 10):
        device = next(model.parameters()).device

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()

                optimizer.zero_grad()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        print("Finished Training")

    def saveModel(self, folder: str = "outputs/nn") -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.name}.pt")

        torch.save(self.state_dict(), path)
        print(f"[NeuralNetwork] Model saved to {path}")
        return path

    @staticmethod
    def loadModel(input_size: int, output_size: int, path: str, device: torch.device) -> nn.Module:
        model = NeuralNetwork(input_size=input_size, output_size=output_size).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model loaded from {path}")
        return model

    @staticmethod
    def getModelPredictions(model: nn.Module, test_loader: DataLoader):
        device = next(model.parameters()).device
        model.eval()

        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                probs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5).long()  # threshold at 0.5

                all_probs.append(probs)
                all_predictions.append(predicted)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_probs = torch.cat(all_probs, dim=0)

        return all_predictions, all_probs

    # Evaluate the model.
    @staticmethod
    def testModel(model: nn.Module, test_loader: DataLoader):
        device = next(model.parameters()).device
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

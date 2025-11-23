import os

import torch
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x  # save input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # add skip connection
        out = F.relu(out)
        return out

class CNN(nn.Module):
    def __init__(
        self,
        board_size=15,
        channels=3,
        res_blocks=6,
        hidden_channels=64,   # number of channels in main conv body
        value_hidden=64       # size of hidden layer in value head
    ):
        super().__init__()

        # Initial convolution to expand to hidden_channels
        self.conv = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_channels) for _ in range(res_blocks)]
        )

        # Head - Which sound is most probable to be
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

    # Instantiate a NN on the specified device
    @staticmethod
    def getModel(board_size=15, channels=3, res_blocks=6, device=None):
        if device is None:
            device = CNN.setDevice()

        model = CNN(board_size=board_size, channels=channels, res_blocks=res_blocks).to(device)
        print(model)
        return model

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

    @staticmethod
    def compute_loss(policy_pred, value_pred, policy_target, value_target):
       # need to decide
       return 0

    # SDG - stocastic gradial descent
    # TODO: Change to Adam optimizer
    @staticmethod
    def getDefaultOptimizer(model: nn.Module, lr: float = 1e-2, momentum: float = 0.9):
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    @staticmethod
    def trainModel(model: nn.Module,
                   train_loader: DataLoader,
                   optimizer,
                   epochs: int = 10,
                   scheduler_step_size: int = 5,
                   scheduler_gamma: float = 0.1):
        device = next(model.parameters()).device

        # Reduce the LR by 10% every 5 epochs (e.g., 0.01 -> 0.001 -> 0.0001 -> ...)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for states, (policy_target, value_target) in train_loader:
                # Move to device
                states = states.to(device)
                policy_target = policy_target.to(device).float()
                value_target = value_target.to(device).float()

                optimizer.zero_grad()

                # Forward pass
                policy_pred, value_pred = model(states)

                loss, policy_loss, value_loss = CNN.compute_loss(policy_pred, value_pred, policy_target, value_target)

                # Backpropagation
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # After the epoch, step the scheduler
            scheduler.step()

            # Print current LR to verify the change
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            print("Policy Loss:", policy_loss.item())
            print("Value Loss:", value_loss.item())


        print("Finished Training")

    def saveModel(self, name: str, folder: str = "../output/") -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name}.pt")

        torch.save(self.state_dict(), path)
        print(f"[CNN] Model saved to {path}")
        return path

    @staticmethod
    def loadModel(input_size: int, output_size: int, path: str, device: torch.device) -> nn.Module:
        model = CNN(input_size=input_size, output_size=output_size).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model loaded from {path}")
        return model

    @staticmethod
    def getModelPredictions(model: nn.Module, test_loader: DataLoader):
        # No need to actually use this function
        device = next(model.parameters()).device
        model.eval()

        all_policy_probs = []
        all_values = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                policy_log_probs, value_pred = model(inputs)

                policy_probs = policy_log_probs.exp()  # convert log_softmax to probabilities
                all_policy_probs.append(policy_probs)
                all_values.append(value_pred)

        all_policy_probs = torch.cat(all_policy_probs, dim=0)
        all_values = torch.cat(all_values, dim=0)

        return all_policy_probs, all_values

    # Evaluate the model.
    @staticmethod
    def testModel(model: nn.Module, test_loader: DataLoader):
        # No need to actually use this function
        device = next(model.parameters()).device
        model.eval()

        total_policy_loss = 0
        total_value_loss = 0
        batches = 0

        with torch.no_grad():
            for inputs, (policy_target, value_target) in test_loader:
                inputs = inputs.to(device)
                policy_target = policy_target.to(device).float()
                value_target = value_target.to(device).float()

                policy_pred, value_pred = model(inputs)

                # compute loss (same as training)
                policy_loss = -torch.sum(policy_target * policy_pred, dim=1).mean()
                value_loss = F.mse_loss(value_pred.squeeze(1), value_target.float())

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                batches += 1

        avg_policy_loss = total_policy_loss / batches
        avg_value_loss = total_value_loss / batches

        print(f"Test Policy Loss: {avg_policy_loss:.4f}, Test Value Loss: {avg_value_loss:.4f}")
        return avg_policy_loss, avg_value_loss

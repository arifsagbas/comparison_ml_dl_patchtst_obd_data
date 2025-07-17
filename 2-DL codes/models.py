import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16  
import numpy as np

class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        return self.model(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class GRUModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

class ViTModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ViTModel, self).__init__()
        self.vit = vit_b_16(weights=None)  # eski 'pretrained=False' karşılığı
        # in_features çekme garantili yol
        if isinstance(self.vit.heads, nn.Sequential):
            in_features = self.vit.heads[0].in_features
        else:
            in_features = self.vit.heads.in_features
        self.vit.heads = nn.Linear(in_features, num_classes)

    def forward(self, x):
        B, T, F = x.shape
        img = x.reshape(B, 1, T, F)  # tek kanallı
        img = nn.functional.interpolate(img, size=(64, 64))  # 🔹 daha küçük input
        img = img.repeat(1, 3, 1, 1)  # sahte RGB
        return self.vit(img)
        
class RNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, 64, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])

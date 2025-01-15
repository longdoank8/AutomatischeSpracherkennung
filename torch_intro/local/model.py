import torch.nn as nn
import torch

class Classification(torch.nn.Module):
    def __init__(self, idim=39, odim=1, hidden_dim=512):
        torch.nn.Module.__init__(self)
        print("Using small model")
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(idim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Linear(hidden_dim, odim)
        self.sigmoid = torch.nn.Sigmoid()     

    def forward(self, audio_feat):
        """
        Vorwärtsauswertung des Netzwerks.
        
        Parameters:
        - audio_feat: Tensor mit Dimension [BS, f_len, idim]
        
        Returns:
        - Tensor mit den posterioren Wahrscheinlichkeiten [BS, odim]
        """
        BS, f_len, idim = audio_feat.shape

        x = audio_feat.reshape(-1, idim)
        x = self.linear_relu_stack(x)
        x = x.reshape(BS, f_len, -1)
        x = torch.mean(x, dim=1)  # [BS, hidden_dim]

        output = self.sigmoid(self.classifier(x))  # [BS, odim]

        return output
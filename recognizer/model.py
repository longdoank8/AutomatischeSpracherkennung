import torch

class DNN_L_Model(torch.nn.Module):
    def __init__(self, idim, odim, hidden_dim):
        print("Using L model")

        torch.nn.Module.__init__(self)
        self.odim = odim
        self.hidden_dim = hidden_dim
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(idim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2*hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, odim)
        )

    def forward(self, audio_feat):  
        audio_feat_flat = torch.flatten(audio_feat, 2, 3).squeeze()
        nn_output = self.linear_relu_stack(audio_feat_flat)

        return nn_output
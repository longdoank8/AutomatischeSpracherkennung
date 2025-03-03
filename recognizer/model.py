import torch

class DNN_Model(torch.nn.Module):
    def __init__(self, idim, odim, hidden_dim):
        print("Using small model")

        torch.nn.Module.__init__(self)
        self.odim = odim
        self.hidden_dim = hidden_dim
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(idim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, odim)
        )

    def forward(self, audio_feat):  
        audio_feat_flat = torch.flatten(audio_feat, 2, 3).squeeze()
        nn_output = self.linear_relu_stack(audio_feat_flat)

        return nn_output


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



class CNN_Model(torch.nn.Module):
    def __init__(self, h, w, idim,  odim, hidden_dim):
        torch.nn.Module.__init__(self)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(16 * 9 * 5, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, odim)
        )

    def forward(self, audio_feat):  
        x = audio_feat.squeeze().unsqueeze(1)
        #print("input shape: ", x.shape)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 16 * 9 * 5)
        x = self.linear_relu_stack(x)
        #print("output shape: ", x.shape)

        return x
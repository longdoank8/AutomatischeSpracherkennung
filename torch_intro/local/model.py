import torch.nn as nn
import torch

class Classification(torch.nn.Module):
    def __init__(self, idim=39, odim=1, hidden_dim=512):
        torch.nn.Module.__init__(self)
        # Define three fully connected layers followed by ReLU activation funtion
        self.fc1 = nn.Linear(idim, hidden_dim)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, odim)
        self.relu3 = nn.ReLU()

        # Define a fully connected output (classification) layer
        self.classifier = nn.Linear(hidden_dim, odim)

        # Sigmoid function computes probability
        self.sigmoid = nn.Sigmoid()        

    def forward(self, audio_feat):
        """
        Input: 
            audio_feat: <tensor.FloatTensor> the audio features in a tensor
        Return: 
            The predicted posterior probabilities
        """
         
        # Features mapped by 3 fully connected layers
        # Average the mapped representation over the sequence dimension
        # Return the predicted probabilities by classification layer and Sigmoid function.
        # Input-Shape 
        BS, f_len, idim = audio_feat.shape

        # Reshape [BS * f_len, idim] 
        x = audio_feat.reshape(-1, idim)

        # map features through complete 3 connected layers
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))

        # Reshape to batch-size and sequence length [BS, f_len, hidden_dim]
        x = x.reshape(BS, f_len, -1)

        # average mapped representation over sequence dimension (f_len): [BS, hidden_dim]
        x = torch.mean(x, dim=1)

        # classification layer and sigmoud [BS, odim]
        output = self.sigmoid(self.classifier(x))

        return output
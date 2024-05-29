import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.layers = nn.Sequential(
      #nn.Flatten(),
      nn.Linear(input_dim, input_dim),
      nn.ReLU(),
      nn.Linear(input_dim, output_dim),
      nn.ReLU(),
      nn.Linear(output_dim, output_dim),
      #nn.ReLU()
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

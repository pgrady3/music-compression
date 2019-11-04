'''
In this model, we apply a CNN based autoencoder to audio data.
Input is assumed to be fixed length vector
'''

import torch
import torch.nn as nn

from collections import OrderedDict


class CNNGenreClassifier(nn.Module):
  '''
  Parent class for model definition
  '''

  def __init__(self, input_size=65536):
    '''
    Initialize the model
    '''
    super().__init__()

    self.input_size = input_size  # assume input size is 65536 (2^16)

    self.cnn_layers = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv1d(1, 64, kernel_size=512, stride=256, padding=0)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv1d(64, 64, kernel_size=21, stride=1)),
        ('relu2', nn.ReLU(inplace=True)),
        ('maxpool3', nn.MaxPool1d(kernel_size=4, stride=4)),
        ('conv4', nn.Conv1d(64, 64, kernel_size=21, stride=1)),
        ('relu4', nn.ReLU(inplace=True)),
        ('maxpool5', nn.MaxPool1d(kernel_size=4, stride=4)),
        ('conv6', nn.Conv1d(64, 64, kernel_size=3, stride=1)),
        ('relu6', nn.ReLU(inplace=True)),
        ('maxpool7', nn.MaxPool1d(kernel_size=4, stride=4))
    ]))

    self.fc_layers = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(64, 32)),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(32, 16)),
    ]))

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

  def forward(self, inputs):
    '''
    Forward pass implementation

    Args:
      - inputs: The music input data. Dimension: N*D*1
    '''

    return self.fc_layers(self.cnn_layers(inputs).view(inputs.shape[0], -1))

  def loss(self, input_data, output_data, normalize=True):
    '''
    Computes the loss between input and output data
    '''
    if normalize:
      return self.loss_criterion(input_data, output_data)/(input_data.shape[0])

    return self.loss_criterion(input_data, output_data)

  def predict_labels(self, inputs):
    '''
    Predict genre labels
    '''
    raw_scores = self.forward(inputs)

    return torch.argmax(raw_scores, dim=1)

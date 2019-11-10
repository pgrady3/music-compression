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

  def __init__(self,
               input_size=65536,
               init_from_autoencoder_flag=True,
               autoencoder_path='model_checkpoints/cnn_autoencoder_type1/checkpoint.pt'):
    '''
    Initialize the model
    '''
    super().__init__()

    self.input_size = input_size  # assume input size is 65536 (2^16)

    self.encoder_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv1d(1, 64, kernel_size=512, stride=256, padding=0)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv1d(64, 32, kernel_size=1, stride=1)),
        ('relu2', nn.ReLU(inplace=True)),
    ]))

    self.cnn_layers = nn.Sequential(OrderedDict([
        ('conv4', nn.Conv1d(32, 32, kernel_size=5, stride=1)),
        ('relu4', nn.ReLU(inplace=True)),
        ('maxpool5', nn.MaxPool1d(kernel_size=5, stride=5)),
        ('conv6', nn.Conv1d(32, 32, kernel_size=5, stride=1)),
        ('relu6', nn.ReLU(inplace=True)),
        ('maxpool7', nn.MaxPool1d(kernel_size=5, stride=5))
    ]))

    self.fc_layers = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(288, 100)),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100, 8)),
    ]))

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    if init_from_autoencoder_flag:
      self.init_from_autoencoder(autoencoder_path)

  def init_from_autoencoder(self, autoencoder_checkpoint_path):
    '''
    Loads the parameters from autoencoder
    '''
    autoencoder_checkpoint = torch.load(autoencoder_checkpoint_path)

    self.encoder_model[0].weight.requires_grad = True
    self.encoder_model[0].bias.requires_grad = True

    self.encoder_model[2].weight.requires_grad = True
    self.encoder_model[2].bias.requires_grad = True

    self_state = self.state_dict()
    for name, param in autoencoder_checkpoint['model_state_dict'].items():
      if name not in self_state:
        continue
      print('copying params from ', name)
      self_state[name].copy_(param)

  def forward(self, inputs):
    '''
    Forward pass implementation

    Args:
      - inputs: The music input data. Dimension: N*D*1
    '''

    return self.fc_layers(self.cnn_layers(self.encoder_model(inputs)).view(inputs.shape[0], -1))

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

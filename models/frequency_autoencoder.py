'''
In this model, we apply a CNN based autoencoder to audio data.
Input is assumed to be fixed length vector
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict


class FrequencyAutoencoder(nn.Module):
  '''
  Parent class for model definition
  '''

  def __init__(self, input_size=(1025, 129)):
    '''
    Initialize the model
    '''
    self.first = True

    super(FrequencyAutoencoder, self).__init__()

    self.input_size = input_size  # assume input size is 257 x 126

    self.kernel_sizes = [(5, 6), (5, 5)]
    self.strides = [(2, 2), (2, 2)]
    self.filters = [2, 4, 8]

    self.encoder_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(
            self.filters[0], self.filters[1], kernel_size=self.kernel_sizes[0], stride=self.strides[0])),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(
            self.filters[1], self.filters[2], kernel_size=self.kernel_sizes[1], stride=self.strides[1])),
        ('relu2', nn.ReLU(inplace=True)),
    ]))

    self.decoder_model = nn.Sequential(OrderedDict([
        ('convT2', nn.ConvTranspose2d(
            self.filters[2], self.filters[1], kernel_size=self.kernel_sizes[1], stride=self.strides[1])),
        ('reluT2', nn.ReLU(inplace=True)),
        ('convT1', nn.ConvTranspose2d(
            self.filters[1], self.filters[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0])),
    ]))

    self.loss_criterion = nn.MSELoss()

  def forward(self, inputs):
    '''
    Forward pass implementation

    Args:
      - inputs: The music input data. Dimension: N*D
    '''
    encoded_data = self.forward_encoder(inputs)
    decoded_data = self.forward_decoder(encoded_data)

    if self.first:
      self.first = False
      print("Input shape", inputs.shape, "size",
            np.prod(np.array(inputs.shape)))
      print("Latent shape", encoded_data.shape, "size",
            np.prod(np.array(encoded_data.shape)))
      print("Output shape", decoded_data.shape, "size",
            np.prod(np.array(decoded_data.shape)))

    return decoded_data, encoded_data

  def forward_encoder(self, inputs):
    return self.encoder_model(inputs)

  def forward_decoder(self, inputs):
    return self.decoder_model(inputs)


class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()

    fc_size = [8 * 62 * 29, 100, 8]  # Forgive me for I have sinned

    self.fc_1 = nn.Linear(fc_size[0], fc_size[1])
    self.fc_2 = nn.Linear(fc_size[1], fc_size[2])

    self.bn_1 = nn.BatchNorm1d(fc_size[1])
    self.bn_2 = nn.BatchNorm1d(fc_size[2])

    self.loss_criterion = nn.CrossEntropyLoss()

  def forward(self, inputs):
    flattened = inputs.view(inputs.shape[0], -1)
    fc_1 = F.relu(self.bn_1(self.fc_1(flattened)))
    fc_2 = F.relu(self.bn_2(self.fc_2(fc_1)))

    return fc_2

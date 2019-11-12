'''
In this model, we apply a CNN based autoencoder to audio data.
Input is assumed to be fixed length vector
'''

import torch
import torch.nn as nn

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

    self.input_size = input_size  # assume input size is 1025 x 129

    self.kernel_sizes = [(5, 4), (5, 4)]
    self.strides = [(2, 2), (2, 2)]
    self.filters = [1, 4, 8]

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
      print("Input shape", inputs.shape, "size", inputs.size)
      print("Latent shape", encoded_data.shape, "size", encoded_data.size)
      print("Output shape", decoded_data.shape, "size", decoded_data.size)

    return decoded_data

  def forward_encoder(self, inputs):
    return self.encoder_model(inputs)

  def forward_decoder(self, inputs):
    return self.decoder_model(inputs)

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

    self.encoder_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 8, kernel_size=8, stride=1)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(8, 32, kernel_size=16, stride=1)),
        ('relu2', nn.ReLU(inplace=True)),
    ]))

    self.decoder_model = nn.Sequential(OrderedDict([
        ('convT2', nn.ConvTranspose2d(
            32, 8, kernel_size=16, stride=1)),
        ('reluT2', nn.ReLU(inplace=True)),
        ('convT1', nn.ConvTranspose2d(8, 1, kernel_size=8, stride=1)),
    ]))

    self.loss_criterion = nn.L1Loss()

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
      print("In shape", inputs.shape, "latent",
            encoded_data.shape, "out shape", decoded_data.shape)

    return decoded_data

  def forward_encoder(self, inputs):
    return self.encoder_model(inputs)

  def forward_decoder(self, inputs):
    return self.decoder_model(inputs)

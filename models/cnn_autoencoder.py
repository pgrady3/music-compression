'''
In this model, we apply a CNN based autoencoder to audio data.
Input is assumed to be fixed length vector
'''

import torch
import torch.nn as nn

from collections import OrderedDict


class CNNAutoEncoder(nn.Module):
  '''
  Parent class for model definition
  '''

  def __init__(self, input_size=65536):
    '''
    Initialize the model
    '''
    super(CNNAutoEncoder, self).__init__()

    self.input_size = input_size  # assume input size is 65536 (2^16)

    self.encoder_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv1d(1, 256, kernel_size=128, stride=64)),  # size (2^8)
        ('conv2', nn.Conv1d(256, 256, kernel_size=7, stride=2)),  # size (2)
        ('conv3', nn.Conv1d(256, 512, kernel_size=7, stride=2)),  # size (2^4)
        ('conv4', nn.Conv1d(512, 1024, kernel_size=7, stride=2)),  # size (2^4)
    ]))

    # size after this = (N, 512, 8)

    self.decoder_model = nn.Sequential(OrderedDict([
        ('convT4', nn.ConvTranspose1d(
            1024, 512, kernel_size=7, stride=2, output_padding=1)),  # size (2^4)
        ('convT3', nn.ConvTranspose1d(
            512, 256, kernel_size=7, stride=2)),
        ('convT2', nn.ConvTranspose1d(
            256, 256, kernel_size=7, stride=2)),
        ('convT1', nn.ConvTranspose1d(
            256, 1, kernel_size=128, stride=64)),
    ]))

    self.loss_criterion = nn.MSELoss()  # mean squared error loss

  def forward(self, inputs):
    '''
    Forward pass implementation

    Args:
      - inputs: The music input data. Dimension: N*D*1
    '''

    encoded_data = self.forward_encoder(inputs)
    decoded_data = self.forward_decoder(encoded_data)

    return decoded_data

  def loss(self, input_data, output_data):
    '''
    Computes the loss between input and output data
    '''
    return self.loss_criterion(output_data, input_data)

  def forward_encoder(self, inputs):
    return self.encoder_model(inputs)

  def forward_decoder(self, inputs):
    return self.decoder_model(inputs)

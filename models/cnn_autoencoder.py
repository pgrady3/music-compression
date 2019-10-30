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
        ('conv1', nn.Conv1d(1, 64, kernel_size=512, stride=256, padding=0)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv1d(64, 32, kernel_size=1, stride=1)),
        ('relu2', nn.ReLU(inplace=True)),
    ]))

    # size after this = (N, 16, )

    self.decoder_model = nn.Sequential(OrderedDict([
        ('convT2', nn.ConvTranspose1d(
            32, 64, kernel_size=1, stride=1, output_padding=0)),
        ('reluT2', nn.ReLU(inplace=True)),
        ('convT1', nn.ConvTranspose1d(
            64, 1, kernel_size=512, stride=256, output_padding=0)),
    ]))

    self.loss_criterion = nn.MSELoss(
        reduction='sum')  # mean squared error loss

  def forward(self, inputs):
    '''
    Forward pass implementation

    Args:
      - inputs: The music input data. Dimension: N*D*1
    '''

    encoded_data = self.forward_encoder(inputs)
    decoded_data = self.forward_decoder(encoded_data)

    return decoded_data

  def loss(self, input_data, output_data, normalize=True):
    '''
    Computes the loss between input and output data
    '''
    if normalize:
      return self.loss_criterion(output_data, input_data)/(input_data.shape[0]*input_data.shape[2])

    return self.loss_criterion(output_data, input_data)/input_data.shape[2]

    # move to spectral loss
    # input_fft_mag = torch.log(
    #     torch.sum(torch.stft(torch.squeeze(input_data, dim=1), 1024,
    #                          normalized=True)**2, dim=3).view(input_data.shape[0], -1) + 1e-8
    # )
    # output_fft_mag = torch.log(
    #     torch.sum(torch.stft(torch.squeeze(output_data, dim=1), 1024,
    #                          normalized=True)**2, dim=3).view(input_data.shape[0], -1) + 1e-8
    # )

    # if normalize:
    #   return self.loss_criterion(output_fft_mag, input_fft_mag)/(input_fft_mag.shape[0]*input_fft_mag.shape[1])

    # return self.loss_criterion(output_fft_mag, input_fft_mag)/(input_fft_mag.shape[1])

  def forward_encoder(self, inputs):
    return self.encoder_model(inputs)

  def forward_decoder(self, inputs):
    return self.decoder_model(inputs)

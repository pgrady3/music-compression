import torch
import numpy as np

from utils.io import read_mp3
from models.cnn_autoencoder import CNNAutoEncoder


if __name__ == '__main__':
  model = CNNAutoEncoder()

  optimizer = torch.optim.Adam(model.parameters(),
                               lr=1e-3,
                               weight_decay=1e-5)

  _, input_data = read_mp3('data/individual_samples/sample1.mp3')

  # center the input data
  input_data = torch.FloatTensor(input_data[:65536])
  input_data = input_data - torch.mean(input_data)
  input_data = input_data/torch.std(input_data)

  input_data = torch.reshape(input_data, (1, 1, -1))

  #

  optimizer.zero_grad()
  encoded_data = model.forward_encoder(input_data)
  print(encoded_data.shape)
  decoded_data = model.decoder_model(encoded_data)
  print(decoded_data.shape)

  #loss = model.loss(input_data, output_data)

  # loss.backward()
  # optimizer.step()

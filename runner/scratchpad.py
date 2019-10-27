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

  for epoch in range(100000):

    optimizer.zero_grad()
    output_data = model.forward(input_data)

    loss = model.loss(input_data, output_data)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))

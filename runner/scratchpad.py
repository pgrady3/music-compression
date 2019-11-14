import torch

from models.cnn_autoencoder import CNNAutoEncoder
from models.cnn_genre_classifier import CNNGenreClassifier
from utils.io import read_mp3


if __name__ == '__main__':
  model = CNNAutoEncoder()

  _, input_data = read_mp3('data/sample/061/train/061006.mp3')

  input_data = torch.reshape(torch.FloatTensor(input_data[:65536]), (1, 1, -1))

  encoded_data = model.forward_encoder(input_data)
  print(encoded_data.shape)
  decoded_data = model.forward_decoder(encoded_data)
  print(decoded_data.shape)

  # loss = model.loss(input_data, decoded_data, normalize=True)
  # print(loss)

  # # loss.backward()
  # # optimizer.step()

  # model = CNNGenreClassifier(init_from_autoencoder_flag=False)

  # _, input_data = read_mp3('data/sample/061/train/061006.mp3')

  # input_data = torch.reshape(torch.FloatTensor(input_data[:65536]), (1, 1, -1))

  # predicted_labels = model.forward(input_data)
  # print(predicted_labels.shape)

  # # loss.backward()
  # # optimizer.step()

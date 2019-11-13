import torch

# from models.cnn_autoencoder import CNNAutoEncoder


if __name__ == '__main__':
  # model = CNNAutoEncoder()

  input_data = torch.load('/home/jayantabhowmick/workspace/music-compression/data/sample_data/000002.pt')
  print(input_data)
  input_data = torch.reshape(input_data, (1, 1, -1))
  print(input_data)
  # encoded_data = model.forward_encoder(input_data)
  # print(encoded_data.shape)
  # decoded_data = model.forward_decoder(encoded_data)
  # print(decoded_data.shape)

  # loss = model.loss(input_data, decoded_data, normalize=True)
  # print(loss)

  # loss.backward()
  # optimizer.step()

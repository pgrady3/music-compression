import os

import torch.utils

from torch.autograd import Variable

from models.frequency_autoencoder import FrequencyAutoencoder
from dataset_loader.music_loader_stft import MusicLoaderSTFT

import matplotlib.pyplot as plt


class TrainerFrequencyAutoencoder(object):
  '''
  This class makes training the model easier
  '''

  def __init__(self, data_dir, model_dir, batch_size=1, load_from_disk=False, cuda=False):
    self.model_dir = model_dir

    self.model = FrequencyAutoencoder()

    self.cuda = cuda
    if cuda:
      self.model.cuda()

    dataloader_args = {'num_workers': 2, 'pin_memory': True} if cuda else {}

    self.train_dataset = MusicLoaderSTFT(data_dir, split='train')
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                    **dataloader_args)

    self.test_dataset = MusicLoaderSTFT(data_dir, split='test')
    self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                                   **dataloader_args
                                                   )

    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=1e-5,
                                      weight_decay=1e-5)

    self.train_loss_history = []
    self.test_loss_history = []

    # load the model from the disk if it exists
    if os.path.exists(model_dir) and load_from_disk:
      checkpoint = torch.load(os.path.join(self.model_dir, 'checkpoint.pt'))
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      # self.loss_history = checkpoint['loss_history'].tolist()
      # self.epoch = checkpoint['epoch']

  def save_model(self):
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
    }, os.path.join(self.model_dir, 'checkpoint.pt'))

  def train(self, num_epochs):
    self.model.train()
    for epoch_idx in range(num_epochs):
      for batch_idx, batch in enumerate(self.train_loader):
        if self.cuda:
          input_data = Variable(batch).cuda()
        else:
          input_data = Variable(batch)

        self.optimizer.zero_grad()
        output_data = self.model.forward(input_data)
        loss = self.model.loss_criterion(input_data, output_data)
        loss.backward()
        self.optimizer.step()

      print('Epoch:{}, Loss:{:.4f}'.format(epoch_idx+1, float(loss)))
      self.train_loss_history.append(float(loss))
      self.eval_on_test()
      self.model.train()
      self.save_model()

  def eval_on_test(self):
    self.model.eval()

    test_loss = 0.0

    num_examples = 0
    for batch_idx, batch in enumerate(self.test_loader):
      if self.cuda:
        input_data = Variable(batch).cuda()
      else:
        input_data = Variable(batch)

      num_examples += input_data.shape[0]

      output_data = self.model.forward(input_data)
      loss = self.model.loss_criterion(input_data, output_data)
      test_loss += float(loss)

    self.test_loss_history.append(test_loss/num_examples)

    return self.test_loss_history[-1]

  def plot_loss_history(self, mode='train'):
    plt.figure()
    if mode == 'train':
      plt.plot(self.train_loss_history)
    elif mode == 'test':
      plt.plot(self.test_loss_history)
    plt.show()


if __name__ == '__main__':
  trainer = TrainerFrequencyAutoencoder(
      'data/fma_xs/', 'model_logs/', cuda=True)

  trainer.train(num_epochs=100)

# run in main project directory with python -m runner.train_frequency_autoencoder

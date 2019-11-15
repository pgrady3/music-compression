import os

import torch.utils

from torch.autograd import Variable

from models.cnn_genre_classifier import CNNGenreClassifier
from dataset_loader.music_genre_loader import MusicGenreLoader
from dataset_loader.music_genre_loader_with_voting import MusicGenreLoaderWithVoting

import matplotlib.pyplot as plt

from scipy.stats import mode


class TrainerClassifier(object):
  '''
  This class makes training the model easier
  '''

  def __init__(self, data_dir, model_dir, batch_size=100, load_from_disk=True, cuda=False, num_votes=None):
    self.model_dir = model_dir
    self.num_votes = num_votes
    self.model = CNNGenreClassifier(init_from_autoencoder_flag=True)

    self.cuda = cuda
    if cuda:
      self.model.cuda()

    dataloader_args = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    self.train_dataset = MusicGenreLoader(data_dir, split='train', snippet_size=65536*2)
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                    **dataloader_args)

    self.test_dataset = MusicGenreLoader(data_dir, split='test', snippet_size=65536*2)
    
    self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                                   **dataloader_args
                                                   )

    if self.num_votes is not None:
      self.test_dataset_voting = MusicGenreLoaderWithVoting(data_dir, split='test', snippet_size=65536*2, num_votes=self.num_votes)
      self.test_loader_voting = torch.utils.data.DataLoader(self.test_dataset_voting, batch_size=batch_size, shuffle=True,
                                                     **dataloader_args
                                                     )

    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=1e-3,
                                      weight_decay=1e-5)

    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
    print(self.optimizer)
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
          input_data, target_data = Variable(
              batch[0]).cuda(), Variable(batch[1]).cuda()
        else:
          input_data, target_data = Variable(batch[0]), Variable(batch[1])

        output_data = self.model.forward(input_data)
        self.optimizer.zero_grad()
        loss = self.model.loss(output_data, target_data)
        loss.backward()
        self.optimizer.step()

      self.lr_scheduler.step()
      if epoch_idx % 1 == 0:
        print('Epoch:{}, Loss:{:.4f}'.format(epoch_idx+1, float(loss)))
        self.train_loss_history.append(float(loss))
        self.eval_on_test()
        self.model.train()
        self.save_model()

      # unfreeze previous layers after 100 epochs
      if epoch_idx==20:
        self.model.unfreeze_layers()

  def eval_on_test(self):
    self.model.eval()

    test_loss = 0.0

    num_examples = 0
    for batch_idx, batch in enumerate(self.test_loader):
      if self.cuda:
        input_data, target_data = Variable(
            batch[0]).cuda(), Variable(batch[1]).cuda()
      else:
        input_data, target_data = Variable(batch[0]), Variable(batch[1])

      num_examples += input_data.shape[0]
      output_data = self.model.forward(input_data)
      loss = self.model.loss(output_data, target_data, normalize=False)

      test_loss += float(loss)

    self.test_loss_history.append(test_loss/num_examples)

    return self.test_loss_history[-1]

  def get_accuracy(self):
    '''
    Get the accuracy on the test dataset
    '''
    self.model.eval()

    num_examples = 0
    num_correct = 0
    for batch_idx, batch in enumerate(self.test_loader):
      if self.cuda:
        input_data, target_data = Variable(
            batch[0]).cuda(), Variable(batch[1]).cuda()
      else:
        input_data, target_data = Variable(batch[0]), Variable(batch[1])

      num_examples += input_data.shape[0]
      output_data = self.model.forward(input_data)
      predicted_labels = torch.argmax(output_data, dim=1)
      num_correct += torch.sum(predicted_labels == target_data).cpu().item()

    return float(num_correct)/float(num_examples)

  def get_accuracy_voting(self):
    '''
    Get the accuracy on the test dataset
    '''
    self.model.eval()

    num_examples = 0
    num_correct = 0
    for batch_idx, batch in enumerate(self.test_loader_voting):
      if self.cuda:
        input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
      else:
        input_data, target_data = Variable(batch[0]), Variable(batch[1])

      input_data = input_data.reshape(input_data.shape[0] * self.num_votes, 1, 
        int(input_data.shape[-1]/self.num_votes))
      num_examples += int(input_data.shape[0]/self.num_votes)
      output_data = self.model.forward(input_data)
      predicted_labels = torch.argmax(output_data, dim=1)
      predicted_labels = predicted_labels.reshape(-1, self.num_votes)
      predicted_labels = torch.tensor(mode(predicted_labels.cpu(), axis=1)[0])

      num_correct += torch.sum(predicted_labels == target_data.cpu()).cpu().item()

    return float(num_correct)/float(num_examples)

  def plot_loss_history(self, mode='train'):
    plt.figure()
    if mode == 'train':
      plt.plot(self.train_loss_history)
    elif mode == 'test':
      plt.plot(self.test_loss_history)
    plt.show()

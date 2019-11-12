import os
import torch
import torch.utils
from torch.autograd import Variable
from models.frequency_autoencoder import FrequencyAutoencoder, Classifier
from dataset_loader.music_loader_stft import MusicLoaderSTFT

import matplotlib.pyplot as plt


class TrainerFrequencyAutoencoder(object):
  '''
  This class makes training the model easier
  '''

  def __init__(self, data_dir, model_dir, batch_size=16, load_from_disk=False, cuda=False):
    self.model_dir = model_dir

    self.model = FrequencyAutoencoder()
    self.model_classifier = Classifier()

    self.cuda = cuda
    if cuda:
      self.model.cuda()
      self.model_classifier.cuda()

    dataloader_args = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    self.train_dataset = MusicLoaderSTFT(data_dir, split='train')
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                    **dataloader_args)

    self.test_dataset = MusicLoaderSTFT(data_dir, split='test')
    self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                                   **dataloader_args
                                                   )

    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=1e-4,
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
      running_loss = 0.0
      num_examples = 0

      for batch, labels in self.train_loader:
        if self.cuda:
          input_data = Variable(batch).cuda()
        else:
          input_data = Variable(batch)

        self.optimizer.zero_grad()
        output_data, latent_data = self.model.forward(input_data)
        loss = self.model.loss_criterion(input_data, output_data)
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item() * input_data.size(0)
        num_examples += input_data.shape[0]

      print('Epoch:{}, Loss:{:.4f}'.format(
          epoch_idx+1, running_loss / num_examples))
      self.train_loss_history.append(float(loss))
      self.eval_on_test()
      self.model.train()
      self.save_model()

  def train_classifier(self, num_epochs):
    self.model.eval()
    for params in self.model.parameters():
      params.requires_grad = False

    self.model_classifier.train()

    for epoch_idx in range(num_epochs):
      running_loss = 0.0
      num_examples = 0

      for batch, labels in self.train_loader:
        if self.cuda:
          input_data = Variable(batch).cuda()
          label_data = Variable(labels).cuda()
        else:
          input_data = Variable(batch)
          label_data = Variable(labels)

        self.optimizer.zero_grad()
        output_data, latent_data = self.model.forward(input_data)
        class_output = self.model_classifier.forward(latent_data)

        loss = self.model_classifier.loss_criterion(class_output, label_data)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item() * input_data.size(0)
        num_examples += input_data.shape[0]

      print('Epoch:{}, Loss:{:.4f}'.format(
          epoch_idx+1, running_loss / num_examples))
      self.train_loss_history.append(float(loss))
      self.eval_classifier()
      self.model_classifier.train()

  def eval_on_test(self):
    self.model.eval()

    test_loss = 0.0

    num_examples = 0
    for batch, labels in self.test_loader:
      if self.cuda:
        input_data = Variable(batch).cuda()
      else:
        input_data = Variable(batch)

      num_examples += input_data.shape[0]

      output_data, latent = self.model.forward(input_data)
      loss = self.model.loss_criterion(input_data, output_data)
      test_loss += float(loss) * input_data.shape[0]

    self.test_loss_history.append(test_loss/num_examples)

    print('Test loss:{:.4f}'.format(test_loss/num_examples))

    return self.test_loss_history[-1]

  def eval_classifier(self):
    self.model.eval()
    self.model_classifier.eval()

    running_loss = 0.0
    num_examples = 0
    num_correct = 0

    for batch, labels in self.test_loader:

      if self.cuda:
        input_data = Variable(batch).cuda()
        label_data = Variable(labels).cuda()
      else:
        input_data = Variable(batch)
        label_data = Variable(labels)

      output_data, latent_data = self.model.forward(input_data)
      class_output = self.model_classifier.forward(latent_data)
      loss = self.model_classifier.loss_criterion(class_output, label_data)
      running_loss += loss.item() * input_data.size(0)
      num_examples += input_data.shape[0]
      _, preds = torch.max(class_output, 1)
      num_correct += torch.sum(preds == label_data.data).cpu().data.numpy()

    self.test_loss_history.append(running_loss/num_examples)

    print('Test loss:{:.4f}, accuracy {:.4f}'.format(
        running_loss/num_examples, num_correct/num_examples))

    return self.test_loss_history[-1]

  def plot_loss_history(self, mode='train'):
    plt.figure()
    if mode == 'train':
      plt.plot(self.train_loss_history)
    elif mode == 'test':
      plt.plot(self.test_loss_history)
    plt.show()

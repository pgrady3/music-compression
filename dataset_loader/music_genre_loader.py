'''
Script with Pytorch's dataloader class
'''
import csv
import os
import glob
import random

import torch
import torch.utils.data as data

from utils.io import convert_audio_to_tensors


class MusicGenreLoader(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self, root_dir, split='train', snippet_size=65536, transform=None):
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

    self.snippet_size = snippet_size

    self.file_to_labels = dict()
    for row in csv.reader(open(os.path.join(self.root, 'file_to_labels.csv'), 'r')):
      k, v = row
      self.file_to_labels[int(k)] = int(v)

    if split == 'train':
      self.curr_folder = self.train_folder
    elif split == 'test':
      self.curr_folder = self.test_folder

    # convert all mp3 files to torch tensors
    # TODO: optimize the conversion
    if len(glob.glob(os.path.join(root_dir, self.curr_folder, '*.pt'))) == 0:
      convert_audio_to_tensors(os.path.join(root_dir, self.curr_folder))

    self.file_names = glob.glob(
        os.path.join(root_dir, self.curr_folder, '*.pt')
    )

    # generate the label maps
    self.label_map = torch.LongTensor([self.file_to_labels[int(os.path.basename(
        fname).split('.')[0])] for fname in self.file_names])

    self.data_len = len(self.file_names)

    self.mean_val = torch.load(os.path.join(root_dir, 'mean.pt'))
    self.std_val = torch.load(os.path.join(root_dir, 'std.pt'))

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    raw_audio = torch.load(self.file_names[index])

    # randomly select a portion from the raw_audio
    start_idx = int((random.randint(0, raw_audio.numel()-self.snippet_size-1)/100.0)*100)


    return (
        ((raw_audio[start_idx:start_idx+self.snippet_size] -
          self.mean_val)/self.std_val).reshape(1, -1),
        self.label_map[index]
    )

  def __len__(self):
    return self.data_len

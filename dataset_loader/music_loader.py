'''
Script with Pytorch's dataloader class
'''
import os
import glob

import torch
import torch.utils.data as data


class MusicLoader(data.Dataset):
  '''
  Class for data loading
  '''

  train_folder = 'train'
  test_folder = 'test'

  def __init__(self, root_dir, split='train', transform=None, val_samples=100):
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

    self.train_files = glob.glob(os.path.join(
        root_dir, self.train_folder, '*.pt'
    ))

    self.test_files = glob.glob(os.path.join(
        root_dir, self.test_folder, '*.pt'
    ))

    self.mean_val = torch.load(os.path.join(root_dir, 'mean.pt'))
    self.std_val = torch.load(os.path.join(root_dir, 'std.pt'))

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    if self.split == 'train':
      tensor = (torch.load(
          self.train_files[index]) - self.mean_val)/self.std_val
    elif self.split == 'val':
      raise NotImplementedError
    elif self.split == 'test':
      tensor = (torch.load(
          self.test_files[index]) - self.mean_val)/self.std_val

    return torch.reshape(tensor, (1, -1))

  def __len__(self):
    if self.split == 'train':
      return len(self.train_files)
    elif self.split == 'val':
      raise NotImplementedError
    elif self.split == 'test':
      return len(self.test_files)

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

  def __init__(self, root_dir, split='train', transform=None):
    self.root = os.path.expanduser(root_dir)
    self.transform = transform
    self.split = split

    if split == 'train':
      self.curr_folder = self.train_folder
    elif split == 'test':
      self.curr_folder = self.test_folder

    self.data_concat = torch.load(
        os.path.join(
            root_dir, self.curr_folder, 'concat.pt'
        )
    )

    self.data_len = self.data_concat.shape[0]

    self.mean_val = torch.load(os.path.join(root_dir, 'mean.pt'))
    self.std_val = torch.load(os.path.join(root_dir, 'std.pt'))

    # normalizing the data
    self.data_concat = torch.reshape(
        (self.data_concat - self.mean_val)/self.std_val,
        (self.data_len, 1, -1)
    )

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """

    return self.data_concat[index, :, :]

  def __len__(self):
    return self.data_len

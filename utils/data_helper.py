import glob
import os

import torch

from utils.io import read_mp3


def generate_concat_tensor(base_dir):
  file_names = glob.glob(os.path.join(
      base_dir, '*.pt'
  ))

  file_names = [f for f in file_names if not 'concat' in f]

  data_list = []
  for f in file_names:
    data_list.append(torch.load(f))

  # concatenate everything
  all_data = torch.stack(data_list, dim=0)

  torch.save(all_data, os.path.join(base_dir, 'concat.pt'))


def stat_all_files(base_dir):
  file_names = glob.glob(os.path.join(
      base_dir, 'test', '*.mp3'
  )) + glob.glob(os.path.join(
      base_dir, 'train', '*.mp3'
  ))

  data_list = []
  for f in file_names:
    _, raw_audio = read_mp3(f)
    data_list.append(torch.FloatTensor(raw_audio))

  # concatenate everything
  all_data = torch.cat(data_list)

  mean_val = torch.mean(all_data)
  std_val = torch.std(all_data-mean_val)

  print('Mean val = {}'.format(mean_val))
  print('Std val = {}'.format(std_val))

  torch.save(mean_val, os.path.join(base_dir, 'mean.pt'))
  torch.save(std_val, os.path.join(base_dir, 'std.pt'))


if __name__ == '__main__':
  stat_all_files('data/sample/061/')
  # generate_concat_tensor('data/sample/061/test')

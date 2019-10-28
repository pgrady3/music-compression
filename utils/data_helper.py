import glob
import os

import torch


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


def get_all_files(base_dir):
  file_names = glob.glob(os.path.join(
      base_dir, '*', '*.pt'
  ))

  file_names = [f for f in file_names if not 'concat' in f]

  data_list = []
  for f in file_names:
    data_list.append(torch.load(f))

  # concatenate everything
  all_data = torch.stack(data_list, dim=0)

  mean_val = torch.mean(all_data)
  std_val = torch.std(all_data-mean_val)

  torch.save(mean_val, os.path.join(base_dir, 'mean.pt'))
  torch.save(std_val, os.path.join(base_dir, 'std.pt'))


if __name__ == '__main__':
  get_all_files('data/sample/061/')
  # generate_concat_tensor('data/sample/061/test')

import glob
import os

import torch


def get_all_files(base_dir):
  file_names = glob.glob(os.path.join(
      base_dir, '*', '*.pt'
  ))

  data_list = []
  for f in file_names:
    data_list.append(torch.load(f))

  # concatenate everything
  all_data = torch.stack(data_list, dim=0)

  print(all_data.shape)

  mean_val = torch.mean(all_data)
  std_val = torch.std(all_data-mean_val)

  torch.save(mean_val, os.path.join(base_dir, 'mean.pt'))
  torch.save(mean_val, os.path.join(base_dir, 'std.pt'))

  print(mean_val)
  print(std_val)


if __name__ == '__main__':
  get_all_files('data/sample/061/')

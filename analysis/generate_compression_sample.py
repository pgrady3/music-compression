
from utils.io import read_mp3, write_mp3

from random import sample

import os
import torch


def generate_samples(compression_model,
                     mean_val,
                     std_val,
                     labels_to_file_map,
                     output_dir_base,
                     num_counts=10,
                     is_cuda=False
                     ):

  for label, file_list in labels_to_file_map.items():
    int_counter = 0
    for f in sample(file_list, num_counts):
      _, inp = read_mp3(f)
      if is_cuda:
        inp_tensor = torch.cuda.FloatTensor(inp)
      else:
        inp_tensor = torch.FloatTensor(inp)

      output_tensor = compression_model.forward(
          (inp_tensor.view(1, 1, -1)-mean_val)/std_val)*std_val + mean_val

      write_mp3(os.path.join(output_dir_base, str(label), '{}_input.mp3'.format(int_counter)),
                44100,
                torch.squeeze(inp_tensor.detach())
                )

      write_mp3(os.path.join(output_dir_base, str(label), '{}_output.mp3'.format(int_counter)),
                44100,
                torch.squeeze(output_tensor.detach())
                )

      int_counter += 1

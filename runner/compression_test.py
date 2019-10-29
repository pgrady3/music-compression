import os
import torch

from utils.io import write_mp3


def compress_and_regen(dirname, filename, model, is_cuda):
  '''
  Loads a torch file, and pass it through the model
  Saves the audio files
  '''

  input_val = torch.load(os.path.join(dirname, filename))

  mean_val = torch.load(os.path.join(dirname, 'mean.pt'))
  std_val = torch.load(os.path.join(dirname, 'std.pt'))

  input_val = (input_val - mean_val)/std_val

  if is_cuda:
    input_val = input_val.cuda()

  output_val = model.forward_decoder(
      model.forward_encoder(input_val.view(1, 1, -1))).view((-1))

  write_mp3(os.path.join(dirname, 'model_audio/input.mp3'),
            44100, (input_val*std_val + mean_val).cpu().detach().numpy())
  write_mp3(os.path.join(dirname, 'model_audio/output.mp3'),
            44100, (output_val*std_val + mean_val).cpu().detach().numpy())

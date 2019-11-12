# functions to help read and write audio files


import os
import glob
import pydub
import numpy as np
import torch
import librosa


def read_mp3(file_name, normalized=False):
  '''
  Reads the mp3 file from the disk
  '''
  audio, sampling_rate = librosa.load(file_name, sr=8000)
  return audio


def convert_audio_to_tensors(file_dir):
  '''
  Converts all the MP3 files into tensors of fixed length
  '''

  files = glob.glob(
      os.path.join(file_dir, '*.mp3')
  )

  for i, f in enumerate(files):
    try:
      print("Converting {}/{} {}".format(i, len(files), f))
      audio = read_mp3(f)

      output_name = os.path.join(
          file_dir, os.path.basename(f).split('.')[0] + '.pt'
      )

      torch.save(torch.tensor(audio, dtype=torch.float32), output_name)
    except:
      print("Failed", f)


if __name__ == '__main__':

  files = glob.glob(
      os.path.join('data/fma_xs/test/*.mp3')
  )

  # for f in files:
  #   framerate, audio = read_mp3(f)
  #   print(framerate)
  # print(np.nonzero(audio))

  convert_audio_to_tensors('data/fma_xs/train/')
  convert_audio_to_tensors('data/fma_xs/test/')

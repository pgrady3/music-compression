# functions to help read and write audio files

import pydub
import numpy as np


def read_mp3(file_name, normalized=False):
  '''
  Reads the mp3 file from the disk
  '''
  a = pydub.AudioSegment.from_mp3(file_name)
  y = np.array(a.get_array_of_samples())
  if a.channels == 2:
    y = y.reshape((-1, 2))
  if normalized:
    return a.frame_rate, np.float32(y) / 2**15
  else:
    return a.frame_rate, preprocess_audio(y)


def preprocess_audio(input):
  '''
  Convert to single channel data
  '''
  return np.mean(input, axis=1)


if __name__ == '__main__':
  _, audio = read_mp3('data/individual_samples/sample1.mp3')

  print(np.nonzero(audio))
